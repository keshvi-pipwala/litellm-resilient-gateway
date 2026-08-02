"""Tests for the resilient gateway. All run offline in mock mode — no keys, no cost.

Mirrors the acceptance-criteria style I use on my own projects: assert the
behaviour a user depends on (failover happens, spend is tracked, budget stops
runaway calls), not just that the code runs.
"""

import pytest

from resilient_gateway import ResilientGateway
from resilient_gateway.gateway import BudgetExceeded


@pytest.fixture
def gw():
    # No provider keys in the test env -> gateway comes up in mock mode.
    return ResilientGateway()


def test_starts_in_mock_mode_without_keys(gw):
    assert gw.mock_mode is True


def test_normal_request_served_by_primary(gw):
    r = gw.chat("hello")
    assert r.content
    assert r.fell_back is False
    # Primary in the default chain is an OpenAI model.
    assert "gpt" in r.model_served.lower()


def test_primary_outage_triggers_failover(gw):
    r = gw.chat("hello", simulate_primary_outage=True)
    assert r.fell_back is True
    # Request still succeeded, just from a different provider.
    assert r.content
    assert "gpt" not in r.model_served.lower()


def test_spend_and_usage_are_tracked(gw):
    for _ in range(3):
        gw.chat("count me")
    report = gw.spend_report()
    assert report["num_calls"] == 3
    assert report["total_tokens"] > 0
    assert isinstance(report["spend_by_model"], dict)


def test_budget_guard_refuses_calls_once_exhausted():
    # Zero budget => the very first call is refused before spending anything.
    gw = ResilientGateway(budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        gw.chat("this should be refused")


def test_spend_by_model_aggregates_across_providers(gw):
    gw.chat("primary call")
    gw.chat("failover call", simulate_primary_outage=True)
    models = gw.spend_report()["spend_by_model"]
    # Two different providers should appear once each provider served a request.
    assert len(models) >= 2
