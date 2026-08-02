"""Core gateway: a thin, defensible wrapper around litellm.Router.

Design goals (each maps to a LiteLLM capability the DevRel role calls out):

  * model routing     -> one logical model name ("chat") fronts many providers
  * fallbacks         -> primary degrades, requests fail over automatically
  * cost controls     -> per-request spend is tracked and a budget can hard-stop
  * observability     -> every call is recorded (model served, tokens, cost, fell_back)

The wrapper deliberately owns the spend accounting itself (reading LiteLLM's
computed cost off each response) rather than relying on async callback internals,
so the behaviour is deterministic and unit-testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm
from litellm import Router

litellm.suppress_debug_info = True
# Don't let a slow/blocked network stall a demo; fall back to the local cost map.
litellm.turn_off_message_logging = True


# The logical model group. Order == priority: OpenAI serves first, and if it
# degrades we fail over to Anthropic, then Gemini. Swapping providers is a
# config change here, not a code change at every call site — that is the whole
# point of an LLM gateway.
DEFAULT_PROVIDER_CHAIN: List[Dict[str, str]] = [
    {"deployment": "chat-openai", "model": "openai/gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    {"deployment": "chat-anthropic", "model": "anthropic/claude-3-5-sonnet-20241022", "key_env": "ANTHROPIC_API_KEY"},
    {"deployment": "chat-gemini", "model": "gemini/gemini-1.5-flash", "key_env": "GEMINI_API_KEY"},
]

LOGICAL_MODEL = "chat"


@dataclass
class CallRecord:
    """One request's worth of observability."""

    model_served: str
    tokens: int
    cost_usd: float
    fell_back: bool


@dataclass
class GatewayResult:
    """What a caller gets back: the text plus how it was served."""

    content: str
    model_served: str
    tokens: int
    cost_usd: float
    fell_back: bool


@dataclass
class SpendTracker:
    """Aggregates spend + usage across calls and enforces an optional budget."""

    budget_usd: Optional[float] = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    calls: List[CallRecord] = field(default_factory=list)

    def record(self, record: CallRecord) -> None:
        self.calls.append(record)
        self.total_cost_usd += record.cost_usd
        self.total_tokens += record.tokens

    @property
    def spend_by_model(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in self.calls:
            out[c.model_served] = round(out.get(c.model_served, 0.0) + c.cost_usd, 8)
        return out

    def would_exceed_budget(self) -> bool:
        return self.budget_usd is not None and self.total_cost_usd >= self.budget_usd


class BudgetExceeded(RuntimeError):
    """Raised when a call is refused because the spend budget is used up."""


class ResilientGateway:
    """A one-interface, multi-provider chat gateway with failover + spend tracking."""

    def __init__(
        self,
        provider_chain: Optional[List[Dict[str, str]]] = None,
        num_retries: int = 2,
        cooldown_time: int = 5,
        budget_usd: Optional[float] = None,
    ) -> None:
        self.provider_chain = provider_chain or DEFAULT_PROVIDER_CHAIN
        self.mock_mode = self._detect_mock_mode()
        self.tracker = SpendTracker(budget_usd=budget_usd)
        self.router = self._build_router(num_retries=num_retries, cooldown_time=cooldown_time)

    # -- setup ---------------------------------------------------------------

    def _detect_mock_mode(self) -> bool:
        """Mock mode = no provider keys configured. Lets anyone run the demo cold."""
        return not any(os.getenv(p["key_env"]) for p in self.provider_chain)

    def _build_router(self, num_retries: int, cooldown_time: int) -> Router:
        # Each provider gets its OWN deployment name so the router treats them as
        # a strict priority chain (primary + ordered fallbacks) rather than
        # load-balancing across identical names. The first entry is the primary.
        model_list: List[Dict[str, Any]] = []
        names: List[str] = []
        for i, p in enumerate(self.provider_chain):
            name = LOGICAL_MODEL if i == 0 else f"{LOGICAL_MODEL}-fallback-{i}"
            names.append(name)
            params: Dict[str, Any] = {"model": p["model"]}
            if self.mock_mode:
                # Offline: return a deterministic canned response, tag it with the
                # provider so a caller can see which one served the request.
                params["mock_response"] = f"[mock:{p['deployment']}] response ok"
                params["api_key"] = "sk-mock"
            else:
                params["api_key"] = os.getenv(p["key_env"])
            model_list.append({"model_name": name, "litellm_params": params})

        # fallbacks: if the primary "chat" fails, walk the ordered backup chain.
        return Router(
            model_list=model_list,
            fallbacks=[{LOGICAL_MODEL: names[1:]}],
            num_retries=num_retries,
            cooldown_time=cooldown_time,
            set_verbose=False,
        )

    # -- calling -------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        simulate_primary_outage: bool = False,
        **kwargs: Any,
    ) -> GatewayResult:
        """Send one chat request through the gateway.

        simulate_primary_outage forces the first deployment to fail so you can
        watch failover happen without actually taking a provider down.
        """
        if self.tracker.would_exceed_budget():
            raise BudgetExceeded(
                f"Budget of ${self.tracker.budget_usd:.4f} reached "
                f"(spent ${self.tracker.total_cost_usd:.4f}). Request refused."
            )

        messages = [{"role": "user", "content": prompt}]
        call_kwargs: Dict[str, Any] = {"model": LOGICAL_MODEL, "messages": messages, **kwargs}
        if simulate_primary_outage:
            # LiteLLM's built-in switch to exercise the fallback path deterministically.
            call_kwargs["mock_testing_fallbacks"] = True

        response = self.router.completion(**call_kwargs)

        served = response.model or "unknown"
        primary_model = self.provider_chain[0]["model"].split("/", 1)[-1]
        fell_back = simulate_primary_outage or (served not in primary_model and primary_model not in served)

        cost = float(response._hidden_params.get("response_cost") or 0.0)
        tokens = int(getattr(response, "usage", None).total_tokens) if getattr(response, "usage", None) else 0

        self.tracker.record(
            CallRecord(model_served=served, tokens=tokens, cost_usd=cost, fell_back=fell_back)
        )

        return GatewayResult(
            content=response.choices[0].message.content,
            model_served=served,
            tokens=tokens,
            cost_usd=cost,
            fell_back=fell_back,
        )

    # -- reporting -----------------------------------------------------------

    def spend_report(self) -> Dict[str, Any]:
        return {
            "mock_mode": self.mock_mode,
            "total_cost_usd": round(self.tracker.total_cost_usd, 8),
            "total_tokens": self.tracker.total_tokens,
            "num_calls": len(self.tracker.calls),
            "spend_by_model": self.tracker.spend_by_model,
            "budget_usd": self.tracker.budget_usd,
        }
