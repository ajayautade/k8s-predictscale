# ============================================
# K8s PredictScale - Shared Test Fixtures
# ============================================

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_metrics_df():
    """Create a realistic metrics DataFrame for testing.

    Generates 200 rows of simulated Kubernetes metrics with
    60-second intervals.
    """
    n = 200
    timestamps = pd.date_range("2026-04-01", periods=n, freq="1min")

    np.random.seed(42)
    base_cpu = 0.3 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = np.random.normal(0, 0.02, n)

    df = pd.DataFrame(
        {
            "cpu_usage": base_cpu + noise,
            "memory_usage": 500e6 + np.random.normal(0, 20e6, n),
            "request_rate": 100 + 30 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 5, n),
            "response_latency_p99": 0.05 + np.random.exponential(0.01, n),
            "error_rate": np.random.poisson(0.5, n).astype(float),
            "ready_replicas": np.full(n, 3.0),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def small_metrics_df():
    """A minimal DataFrame for quick unit tests (20 rows)."""
    n = 20
    timestamps = pd.date_range("2026-04-01", periods=n, freq="1min")

    np.random.seed(123)
    df = pd.DataFrame(
        {
            "cpu_usage": np.random.uniform(0.2, 0.8, n),
            "memory_usage": np.random.uniform(200e6, 800e6, n),
            "request_rate": np.random.uniform(50, 200, n),
            "response_latency_p99": np.random.uniform(0.01, 0.1, n),
            "error_rate": np.random.uniform(0, 2, n),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def sample_prediction_arrays():
    """Sample X, y arrays mimicking pipeline output."""
    np.random.seed(99)
    X = np.random.rand(50, 60, 8).astype(np.float32)
    y = np.random.rand(50, 10).astype(np.float32)
    return X, y
