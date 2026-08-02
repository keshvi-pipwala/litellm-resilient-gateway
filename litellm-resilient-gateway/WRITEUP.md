# Stop hand-swapping LLM providers: automatic failover with LiteLLM

*A short technical walkthrough — the kind of runnable, problem-first content I'd
write for developers adopting LiteLLM.*

## The moment this solves

I was running a PR-review agent on the Claude API. Under load, it started
rate-limiting. My options in the moment were bad: let requests fail, or do what I
actually did — stop, change the provider in code, and redeploy to Gemini. It
worked, but I was hand-operating a production system at the worst possible time,
and I had no clean view of what the swap did to my spend.

The general shape of this problem is everywhere once you run LLMs in production:
providers have outages, rate limits, and latency spikes, and no single one is
reliable enough to bet an app on. You want the *reliability* of many providers
without writing per-provider branching at every call site. That is what an LLM
gateway gives you, and it's what LiteLLM's `Router` does in a few lines.

## The fix in one object

The instinct is to treat each provider as a separate deployment and let the
router walk them in priority order:

```python
from litellm import Router

router = Router(
    model_list=[
        {"model_name": "chat",            "litellm_params": {"model": "openai/gpt-4o-mini"}},
        {"model_name": "chat-fallback-1", "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}},
        {"model_name": "chat-fallback-2", "litellm_params": {"model": "gemini/gemini-1.5-flash"}},
    ],
    fallbacks=[{"chat": ["chat-fallback-1", "chat-fallback-2"]}],
    num_retries=2,
    cooldown_time=5,
)

resp = router.completion(model="chat", messages=[{"role": "user", "content": "hi"}])
```

Callers only ever ask for `"chat"`. If OpenAI errors, LiteLLM retries, then fails
over to Anthropic, then Gemini — no branching in the caller, no redeploy.

## The gotcha worth writing down

My first version gave **all three** deployments the same `model_name: "chat"`.
That is a different feature: identical names make the router *load-balance*
across them, so a "normal" request came back from Anthropic and looked like it
had failed over when nothing had failed. The failover semantics I wanted require
**distinct** deployment names plus an explicit `fallbacks` map.

This is exactly the kind of subtlety that belongs in docs and examples: the two
patterns (load-balancing vs. priority failover) look almost identical in config
but behave very differently, and the difference is easy to get wrong on a first
read. A good example makes it obvious which one you're getting.

## Prove it without spending anything

You don't need live keys or a real outage to demonstrate failover. LiteLLM has
two testing hooks that make the behaviour deterministic:

- `mock_response` in `litellm_params` returns a canned response — so the whole
  demo runs offline, and I can tag each provider's mock so you can *see* which
  one served a request.
- `mock_testing_fallbacks=True` on the call forces the primary to fail, so you
  can watch the fallback chain fire on demand.

That combination is what lets this repo run cold for a new developer — `python
demo.py`, no keys — and still show a real failover happening. Removing the "set
up three API keys before you can see anything" barrier is, in my experience, the
difference between an example people run and one they skim.

## Then make the invisible visible

The other half of the manual-swap pain was that I couldn't see the cost impact.
LiteLLM computes a per-response cost, so the gateway can aggregate spend per model
and even enforce a budget:

```python
cost = response._hidden_params.get("response_cost")   # per-call, computed by LiteLLM
usage = response.usage.total_tokens
```

Roll those up and you get a spend-by-model report and a hard budget stop for free
— the observability I was missing the first time around.

## Takeaways for anyone adopting LiteLLM

1. **Distinct deployment names + a `fallbacks` map = priority failover.** Same
   name = load balancing. Pick deliberately.
2. **Make examples runnable with zero setup** using `mock_response` and
   `mock_testing_fallbacks`. Failover you can *see* offline beats a paragraph
   describing it.
3. **Track spend from day one** off `response_cost` — the gateway is the natural
   place to enforce budgets.

The full runnable version — gateway, three-scenario demo, and six offline tests —
is in this repo. Clone it, run `python demo.py`, and you'll watch a provider
"outage" fail over in about two seconds without touching a key.

---

*Written by Keshvi Pipwala. Code and claims verified against LiteLLM 1.95.*
