"""
OpenTelemetry instrumentation for Axera.

Instruments inference and training calls with spans and metrics
compatible with Jaeger, Grafana, and Datadog.

When the ``opentelemetry-sdk`` package is not installed, all calls
are no-ops — telemetry is completely optional.

Usage
-----
    from axera.telemetry import setup_telemetry, trace_span

    setup_telemetry(service_name="my-clinical-model", endpoint="http://localhost:4317")

    with trace_span("inference") as span:
        span.set_attribute("batch_size", 32)
        predictions = model.predict(X)
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger(__name__)

# ── Optional OTel import ──────────────────────────────────────────────────────

try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


_tracer = None
_meter  = None


def setup_telemetry(
    service_name: str = "axera",
    endpoint: Optional[str] = None,
    insecure: bool = True,
) -> None:
    """
    Initialise OpenTelemetry tracing and metrics.

    Parameters
    ----------
    service_name : str
        Resource name shown in Jaeger / Grafana.
    endpoint : str, optional
        OTLP gRPC endpoint, e.g. ``"http://localhost:4317"``.
        If None, a console exporter is used.
    insecure : bool
        Skip TLS verification for local dev (default True).
    """
    global _tracer, _meter

    if not _OTEL_AVAILABLE:
        logger.warning("opentelemetry-sdk not installed — telemetry disabled.")
        return

    resource = Resource.create({"service.name": service_name})

    # Trace provider
    tp = TracerProvider(resource=resource)
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            tp.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
            ))
        except ImportError:
            logger.warning("OTLP exporter not available; using console.")
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(tp)
    _tracer = trace.get_tracer(service_name)

    # Meter provider
    mp = MeterProvider(resource=resource)
    otel_metrics.set_meter_provider(mp)
    _meter = otel_metrics.get_meter(service_name)

    logger.info("OpenTelemetry initialised for service '%s'", service_name)


@contextlib.contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Context manager that wraps code in an OTel span.

    Falls back to a no-op context if OTel is not configured.
    """
    if _tracer is None:
        yield _NoOpSpan()
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


def instrument(name: Optional[str] = None):
    """
    Decorator that wraps a function in an OTel span.

    Usage::

        @instrument("my_function")
        def my_function(x):
            return heavy_computation(x)
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            with trace_span(span_name) as span:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("duration_ms", round(elapsed_ms, 2))
                return result

        return wrapper

    return decorator


class _NoOpSpan:
    """Fallback span when OTel is not configured."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ANN401
        pass

    def add_event(self, name: str, **kwargs: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass


__all__ = ["setup_telemetry", "trace_span", "instrument"]
