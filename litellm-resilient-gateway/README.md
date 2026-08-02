# Resilient LLM Gateway (built on LiteLLM)

**One interface. Many providers. Automatic failover. Tracked spend.**

A small, production-shaped demo of the LiteLLM value proposition: call any
provider through a single interface, fail over automatically when one degrades,
and track cost per request. It runs **cold, with zero API keys** (mock mode), and
against **live providers** the moment you add keys — no code change.

```
$ python demo.py
====================================================================
  RESILIENT LLM GATEWAY  ·  built on LiteLLM
  mode: MOCK (no API keys found — running offline)
====================================================================

[1] Normal request  → should be served by the primary (OpenAI)
    served by : gpt-4o-mini
    fell back : False

[2] Primary provider is down  → gateway fails over automatically
    served by : claude-3-5-sonnet-20241022
    fell back : True   <- request still succeeded

[3] Cost tracking + budget guard
    calls           : 5
    total tokens    : 150
    total spend     : $0.000054
    spend by model  : {'gpt-4o-mini': 5.4e-05, 'claude-3-5-sonnet-20241022': 0.0}
    budget          : $0.05
```

## Why I built this

On a prior project ([GitSense](https://github.com/keshvi-pipwala/gitsense), a
PR-review agent), the Claude API started rate-limiting me under load, so I
**manually** swapped the app over to Gemini and owned the cost/latency/quality
tradeoff. It worked — but it was a hand-operation at exactly the wrong moment.

That is the problem LiteLLM removes. This repo is the "what I *should* have built"
version: the provider swap becomes automatic and invisible to callers, and I get
spend visibility for free. I built it to learn LiteLLM the way a developer
adopting it actually would — and to have a runnable reference I can point people
to.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py          # runs offline in mock mode, zero keys, zero cost
python -m pytest -q     # 6 tests, all offline
```

To run against real providers, set any subset of keys (it routes to what's
available and fails over across the rest):

```bash
export OPENAI_API_KEY=...      # primary
export ANTHROPIC_API_KEY=...   # first fallback
export GEMINI_API_KEY=...      # second fallback
python demo.py                 # mode: LIVE
```

## What it demonstrates (mapped to LiteLLM capabilities)

| Capability      | Where it lives                                            |
| --------------- | --------------------------------------------------------- |
| Model routing   | one logical name `chat` fronts a priority chain of providers |
| Fallbacks       | `Router(fallbacks=...)` walks the chain when the primary fails |
| Cost controls   | per-request spend is tracked; an optional budget hard-stops calls |
| Observability   | every call records model served, tokens, cost, and whether it fell back |
| Retries/cooldown| `num_retries` + `cooldown_time` on the router             |

## How it works

`resilient_gateway/gateway.py` wraps `litellm.Router`. Each provider is a
separate deployment so the router treats them as a strict **primary + ordered
fallback** chain rather than load-balancing identical names:

```python
Router(
    model_list=[
        {"model_name": "chat",            "litellm_params": {"model": "openai/gpt-4o-mini"}},
        {"model_name": "chat-fallback-1", "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}},
        {"model_name": "chat-fallback-2", "litellm_params": {"model": "gemini/gemini-1.5-flash"}},
    ],
    fallbacks=[{"chat": ["chat-fallback-1", "chat-fallback-2"]}],
    num_retries=2,
    cooldown_time=5,
)
```

The wrapper reads LiteLLM's computed `response_cost` and token usage off each
response and aggregates it, so spend accounting is deterministic and unit-tested
(no reliance on async callback internals). `simulate_primary_outage=True` uses
LiteLLM's `mock_testing_fallbacks` to force the failover path on demand.

## Layout

```
resilient_gateway/
  gateway.py      # ResilientGateway + SpendTracker (the core, ~180 lines)
  __init__.py
demo.py           # 3-scenario walkthrough, runs offline
tests/
  test_gateway.py # 6 offline tests: failover, spend tracking, budget guard
requirements.txt
.env.example
WRITEUP.md        # short technical writeup of the failover pattern
```

## Notes & honest limits

- Mock mode returns canned responses so anyone can run it without keys; costs in
  mock mode reflect whatever LiteLLM's local price map computes (some providers
  resolve to `$0.00` offline). With network + real keys, costs are exact.
- This is a focused demo, not a full gateway — it intentionally leaves out
  auth, persistence, and the proxy server. The point is to show the core
  reliability + spend pattern clearly and runnably.

---

Built by Keshvi Pipwala as a hands-on LiteLLM reference. Verified against
LiteLLM 1.95 (Aug 2026).
