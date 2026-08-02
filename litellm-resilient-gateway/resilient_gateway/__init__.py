"""Resilient LLM Gateway — a small, production-shaped demo built on LiteLLM.

Shows the LiteLLM value proposition end to end: call many providers through one
interface, fail over automatically when a provider degrades, and track spend per
request. Runs offline with mock responses (zero API keys) and against real
providers when keys are present.
"""

from .gateway import ResilientGateway, SpendTracker, GatewayResult

__all__ = ["ResilientGateway", "SpendTracker", "GatewayResult"]
__version__ = "0.1.0"
