#!/usr/bin/env python3
# ============================================
# K8s PredictScale - Load Test Script
# ============================================
# Generates HTTP traffic against a target
# application to simulate scaling scenarios
# and validate the prediction pipeline.
# ============================================

import argparse
import asyncio
import math
import time
from datetime import datetime

import httpx


async def send_requests(
    url: str,
    rps: int,
    duration_seconds: int,
    timeout: float = 5.0,
):
    """Send requests at a constant rate.

    Args:
        url: Target URL.
        rps: Requests per second.
        duration_seconds: How long to maintain the load.
        timeout: HTTP timeout per request.
    """
    total = rps * duration_seconds
    interval = 1.0 / rps if rps > 0 else 1.0
    success = 0
    errors = 0
    total_latency = 0.0

    print(f"  Sending {rps} req/s for {duration_seconds}s ({total} total requests)")

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(total):
            start = time.monotonic()
            try:
                response = await client.get(url)
                latency = time.monotonic() - start
                total_latency += latency
                if response.status_code < 500:
                    success += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

            # Pace requests
            elapsed = time.monotonic() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    avg_latency = (total_latency / (success or 1)) * 1000  # ms
    print(f"  ✅ Done: {success} ok, {errors} errors, avg latency {avg_latency:.1f}ms")
    return {"success": success, "errors": errors, "avg_latency_ms": avg_latency}


async def ramp_load(
    url: str,
    start_rps: int,
    end_rps: int,
    step_duration: int = 30,
    steps: int = 5,
):
    """Gradually ramp load from start_rps to end_rps.

    This simulates a traffic spike that PredictScale should detect and
    pre-scale for.
    """
    print(f"\n{'='*60}")
    print(f"🔥 Load Test: Ramp {start_rps} → {end_rps} req/s over {steps} steps")
    print(f"   Target: {url}")
    print(f"   Step duration: {step_duration}s")
    print(f"{'='*60}\n")

    rps_step = (end_rps - start_rps) / max(steps - 1, 1)
    results = []

    for i in range(steps):
        current_rps = int(start_rps + rps_step * i)
        print(f"\n📈 Step {i+1}/{steps}: {current_rps} req/s")
        result = await send_requests(url, current_rps, step_duration)
        results.append({"step": i + 1, "rps": current_rps, **result})

    print(f"\n{'='*60}")
    print("📊 Load Test Summary")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  Step {r['step']}: {r['rps']:>4} req/s  |  "
            f"✅ {r['success']:>5}  ❌ {r['errors']:>3}  |  "
            f"⏱  {r['avg_latency_ms']:.1f}ms"
        )


async def spike_load(
    url: str,
    baseline_rps: int = 10,
    spike_rps: int = 100,
    baseline_duration: int = 60,
    spike_duration: int = 120,
):
    """Simulate a sudden traffic spike.

    Pattern: baseline → sudden spike → baseline
    """
    print(f"\n{'='*60}")
    print(f"⚡ Spike Test: {baseline_rps} → {spike_rps} → {baseline_rps} req/s")
    print(f"   Target: {url}")
    print(f"{'='*60}\n")

    print("📊 Phase 1: Baseline")
    await send_requests(url, baseline_rps, baseline_duration)

    print("\n🔥 Phase 2: SPIKE!")
    await send_requests(url, spike_rps, spike_duration)

    print("\n📉 Phase 3: Return to baseline")
    await send_requests(url, baseline_rps, baseline_duration)


def main():
    parser = argparse.ArgumentParser(description="Load test for PredictScale validation")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Target URL")
    parser.add_argument(
        "--mode", type=str, choices=["ramp", "spike", "constant"], default="ramp",
        help="Load test mode",
    )
    parser.add_argument("--rps", type=int, default=10, help="Requests per second (constant mode)")
    parser.add_argument("--start-rps", type=int, default=10, help="Starting RPS (ramp mode)")
    parser.add_argument("--end-rps", type=int, default=100, help="Ending RPS (ramp mode)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--steps", type=int, default=5, help="Number of ramp steps")
    args = parser.parse_args()

    if args.mode == "ramp":
        asyncio.run(ramp_load(args.url, args.start_rps, args.end_rps, args.duration, args.steps))
    elif args.mode == "spike":
        asyncio.run(spike_load(args.url, baseline_rps=args.start_rps, spike_rps=args.end_rps))
    else:
        asyncio.run(send_requests(args.url, args.rps, args.duration))


if __name__ == "__main__":
    main()
