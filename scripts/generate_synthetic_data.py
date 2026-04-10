#!/usr/bin/env python3
# ============================================
# K8s PredictScale - Synthetic Data Generator
# ============================================
# Generates realistic Kubernetes workload metrics
# for local development and model training without
# requiring a live Prometheus instance.
# ============================================

import argparse
import os
import sys

import numpy as np
import pandas as pd


def generate_daily_pattern(n: int) -> np.ndarray:
    """Simulate a 24-hour traffic pattern with morning/evening peaks."""
    hours = np.linspace(0, 24 * (n / 1440), n)  # 1440 minutes per day
    # Morning peak at ~10am, evening peak at ~8pm, low at ~4am
    pattern = (
        0.3
        + 0.25 * np.sin(2 * np.pi * (hours - 6) / 24)
        + 0.15 * np.sin(4 * np.pi * (hours - 3) / 24)
    )
    return np.clip(pattern, 0.05, 1.0)


def generate_weekly_pattern(n: int) -> np.ndarray:
    """Add a weekly seasonality (lower on weekends)."""
    days = np.arange(n) / 1440.0
    weekly = 1.0 - 0.2 * np.where(
        (days % 7 >= 5), 1.0, 0.0  # Saturday=5, Sunday=6
    )
    return weekly


def generate_spikes(n: int, n_spikes: int = 5, seed: int = 42) -> np.ndarray:
    """Generate random traffic spikes."""
    rng = np.random.RandomState(seed)
    spikes = np.zeros(n)
    for _ in range(n_spikes):
        center = rng.randint(100, n - 100)
        width = rng.randint(10, 60)
        height = rng.uniform(0.3, 0.8)
        spike = height * np.exp(-0.5 * ((np.arange(n) - center) / width) ** 2)
        spikes += spike
    return spikes


def generate_metrics(
    duration_hours: int = 168,
    interval_minutes: int = 1,
    base_replicas: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a full synthetic metrics DataFrame.

    Args:
        duration_hours: Total hours of data to generate.
        interval_minutes: Time step between data points.
        base_replicas: Baseline pod count.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns matching the collector's output.
    """
    rng = np.random.RandomState(seed)
    n = int(duration_hours * 60 / interval_minutes)
    timestamps = pd.date_range(
        end=pd.Timestamp.utcnow(),
        periods=n,
        freq=f"{interval_minutes}min",
    )

    # Base patterns
    daily = generate_daily_pattern(n)
    weekly = generate_weekly_pattern(n)
    spikes = generate_spikes(n, n_spikes=8, seed=seed)
    combined = daily * weekly + spikes

    # ---- CPU usage ----
    cpu = combined * 0.7 + rng.normal(0, 0.02, n)
    cpu = np.clip(cpu, 0.01, 2.0)

    # ---- Memory usage ----
    memory_base = 400e6 + combined * 200e6
    memory = memory_base + rng.normal(0, 20e6, n)
    memory = np.clip(memory, 100e6, 2000e6)

    # ---- Request rate ----
    request_rate = combined * 200 + rng.normal(0, 10, n)
    request_rate = np.clip(request_rate, 0, 1000)

    # ---- Response latency (p99) ----
    latency_base = 0.05 + combined * 0.03
    latency = latency_base + rng.exponential(0.005, n)
    latency = np.clip(latency, 0.005, 1.0)

    # ---- Error rate ----
    error_rate = rng.poisson(0.3, n).astype(float)
    # Increase errors during high load
    high_load = combined > 0.7
    error_rate[high_load] += rng.poisson(2, high_load.sum())

    # ---- Network I/O ----
    network = combined * 50e6 + rng.normal(0, 5e6, n)
    network = np.clip(network, 0, 200e6)

    # ---- Replicas ----
    replicas = base_replicas + (combined * 5).astype(int)
    replicas = np.clip(replicas, base_replicas, 20)

    df = pd.DataFrame(
        {
            "cpu_usage": cpu,
            "memory_usage": memory,
            "request_rate": request_rate,
            "response_latency_p99": latency,
            "response_latency_p50": latency * 0.4 + rng.exponential(0.002, n),
            "error_rate": error_rate,
            "network_receive": network,
            "ready_replicas": replicas.astype(float),
            "desired_replicas": replicas.astype(float),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Kubernetes workload metrics"
    )
    parser.add_argument(
        "--hours", type=int, default=168, help="Duration in hours (default: 168 = 7 days)"
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="Interval in minutes (default: 1)"
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic_metrics.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    print(f"Generating {args.hours}h of synthetic metrics (interval={args.interval}m)...")
    df = generate_metrics(
        duration_hours=args.hours,
        interval_minutes=args.interval,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)
    print(f"✅ Saved {len(df)} rows to {args.output}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Time range: {df.index.min()} → {df.index.max()}")


if __name__ == "__main__":
    main()
