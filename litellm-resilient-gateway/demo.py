"""Runnable walkthrough of the resilient gateway.

    python demo.py

Runs three scenarios with zero setup (mock mode). Add real provider keys to
your environment and it exercises the same paths against live APIs — no code
change. This is the whole pitch of an LLM gateway in ~60 lines of output.
"""

from resilient_gateway import ResilientGateway


def line(char: str = "-") -> None:
    print(char * 68)


def main() -> None:
    gw = ResilientGateway(budget_usd=0.05)

    line("=")
    print("  RESILIENT LLM GATEWAY  ·  built on LiteLLM")
    print(f"  mode: {'MOCK (no API keys found — running offline)' if gw.mock_mode else 'LIVE'}")
    line("=")

    # 1) Happy path — served by the primary provider.
    print("\n[1] Normal request  → should be served by the primary (OpenAI)")
    r = gw.chat("Summarize what an LLM gateway does in one sentence.")
    print(f"    served by : {r.model_served}")
    print(f"    fell back : {r.fell_back}")
    print(f"    response  : {r.content}")

    # 2) Primary outage — automatic failover to the next provider.
    print("\n[2] Primary provider is down  → gateway fails over automatically")
    print("    (This is exactly the moment my GitSense project hit manually:")
    print("     Claude API rate-limited, so I hand-swapped to Gemini. A gateway")
    print("     makes that swap automatic and invisible to callers.)")
    r = gw.chat("Same request, but pretend OpenAI is rate-limiting us.",
                simulate_primary_outage=True)
    print(f"    served by : {r.model_served}")
    print(f"    fell back : {r.fell_back}   <- request still succeeded")
    print(f"    response  : {r.content}")

    # 3) A few more calls, then the spend report (cost control + observability).
    print("\n[3] Cost tracking + budget guard")
    for q in ["What is failover?", "What is load balancing?", "What is a cooldown?"]:
        gw.chat(q)
    report = gw.spend_report()
    print(f"    calls           : {report['num_calls']}")
    print(f"    total tokens    : {report['total_tokens']}")
    print(f"    total spend     : ${report['total_cost_usd']:.6f}")
    print(f"    spend by model  : {report['spend_by_model']}")
    print(f"    budget          : ${report['budget_usd']}")

    line("=")
    print("  One interface. Many providers. Automatic failover. Tracked spend.")
    line("=")


if __name__ == "__main__":
    main()
