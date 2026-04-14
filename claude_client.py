"""
claude_client.py — Claude API wrapper with streaming, history, and cost tracking

Wraps the Anthropic SDK to provide:
  - Streaming text generation
  - RAG context injection via system prompt
  - Multi-turn conversation history
  - Token counting and per-call cost estimation
  - Retry logic with exponential back-off on transient errors
"""

import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import anthropic

from config import Config


# Pricing per million tokens (claude-sonnet-4, May 2025)
# Update when Anthropic changes pricing.
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
    "claude-haiku-4-20250514":  {"input": 0.25,  "output": 1.25},
    "_default":                 {"input": 3.00,  "output": 15.00},
}

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful AI assistant that answers questions based on the provided documents.

Instructions:
- Base your answers primarily on the retrieved document context below.
- If the context does not contain enough information to answer fully, say so clearly \
and provide what partial information you can.
- Always cite your sources using the [Source N] labels in the context.
- Be concise and accurate. Do not hallucinate or make up information not in the context.
- If the user asks something unrelated to the documents, answer helpfully from \
your general knowledge but clarify that the documents don't cover this topic.

{context_section}
"""

NO_DOCS_SYSTEM_PROMPT = """\
You are a helpful AI assistant. No documents have been uploaded yet.
Answer questions from your general knowledge and encourage the user to upload \
documents for document-specific Q&A.
"""


@dataclass
class Message:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class UsageStats:
    """Token counts and estimated cost for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "UsageStats") -> "UsageStats":
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass
class ConversationSession:
    """State for a single chat conversation."""

    session_id: str
    history: List[Message] = field(default_factory=list)
    total_usage: UsageStats = field(default_factory=UsageStats)

    def add_user_message(self, text: str) -> None:
        """Append a user message to history."""
        self.history.append(Message(role="user", content=text))

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant message to history."""
        self.history.append(Message(role="assistant", content=text))

    def to_api_messages(self) -> List[dict]:
        """Convert history to the format expected by the Anthropic Messages API."""
        return [{"role": m.role, "content": m.content} for m in self.history]

    def clear(self) -> None:
        """Reset conversation history (keeps session_id and cumulative usage)."""
        self.history.clear()


class ClaudeClient:
    """Anthropic Claude client with RAG context injection and streaming.

    Usage:
        client = ClaudeClient(config)
        for token in client.stream(session, "What is RAG?", context_str):
            print(token, end="", flush=True)
    """

    def __init__(self, config: Config) -> None:
        """Initialise the client.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._pricing = MODEL_PRICING.get(config.claude_model, MODEL_PRICING["_default"])

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def stream(
        self,
        session: ConversationSession,
        user_message: str,
        context: str = "",
        max_retries: int = 3,
    ) -> Generator[str, None, None]:
        """Stream a response from Claude, yielding text tokens as they arrive.

        Appends the user message to session history before calling the API
        and appends the full assistant response after streaming completes.

        Args:
            session: Active conversation session.
            user_message: The user's latest message.
            context: RAG context string (formatted by RAGEngine.build_context).
            max_retries: Number of retry attempts on transient API errors.

        Yields:
            String tokens from the streaming response.
        """
        session.add_user_message(user_message)
        system_prompt = self._build_system_prompt(context)
        messages = session.to_api_messages()

        full_response = ""
        usage = UsageStats()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                with self.client.messages.stream(
                    model=self.config.claude_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=messages,
                ) as stream_ctx:
                    for text in stream_ctx.text_stream:
                        full_response += text
                        yield text

                    # Collect usage after stream closes
                    final_msg = stream_ctx.get_final_message()
                    usage = self._compute_usage(final_msg.usage)
                    break  # Success — exit retry loop

            except anthropic.RateLimitError as e:
                last_error = e
                time.sleep(2 ** attempt)
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code and e.status_code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    break  # Non-retryable (auth error, bad request, etc.)
            except Exception as e:
                last_error = e
                break
        else:
            # All retries exhausted
            error_text = f"\n\n[Error communicating with Claude: {last_error}]"
            full_response += error_text
            yield error_text

        session.add_assistant_message(full_response)
        session.total_usage = session.total_usage + usage

    # ------------------------------------------------------------------
    # Non-streaming (for testing / health-check)
    # ------------------------------------------------------------------

    def complete(
        self,
        session: ConversationSession,
        user_message: str,
        context: str = "",
    ) -> Tuple[str, UsageStats]:
        """Non-streaming completion, returns full response and usage.

        Args:
            session: Active conversation session.
            user_message: The user's message.
            context: RAG context string.

        Returns:
            Tuple of (response_text, UsageStats).
        """
        session.add_user_message(user_message)
        system_prompt = self._build_system_prompt(context)

        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=session.to_api_messages(),
        )

        text = response.content[0].text
        usage = self._compute_usage(response.usage)
        session.add_assistant_message(text)
        session.total_usage = session.total_usage + usage
        return text, usage

    # ------------------------------------------------------------------
    # Token counting (pre-flight estimate)
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Uses tiktoken if available, otherwise falls back to word count * 1.3.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, context: str) -> str:
        """Construct the system prompt with optional RAG context.

        Args:
            context: Retrieved document context string.

        Returns:
            System prompt string.
        """
        if not context:
            return NO_DOCS_SYSTEM_PROMPT

        context_section = (
            "Retrieved document context:\n"
            "================================\n"
            f"{context}\n"
            "================================"
        )
        return SYSTEM_PROMPT_TEMPLATE.format(context_section=context_section)

    def _compute_usage(self, usage_obj: object) -> UsageStats:
        """Convert an Anthropic Usage object to our UsageStats dataclass.

        Args:
            usage_obj: anthropic.types.Usage from the API response.

        Returns:
            Populated UsageStats.
        """
        input_tok = getattr(usage_obj, "input_tokens", 0) or 0
        output_tok = getattr(usage_obj, "output_tokens", 0) or 0

        cost = (
            input_tok / 1_000_000 * self._pricing["input"]
            + output_tok / 1_000_000 * self._pricing["output"]
        )

        return UsageStats(
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
        )
