"""OpenTelemetry wiring for the API: RED (auto), per-stage RAG spans, LLM token and
answer-outcome metrics. Vendor-neutral — exports to a local Prometheus + Tempo, or to
Azure Application Insights when APPLICATIONINSIGHTS_CONNECTION_STRING is set.

Instrumentation lives at the edge: the adapters and answer_question stay OTel-free.
`Traced` proxies the injected ports (like profile_latency's Timed) to emit a span and
stage latency per call; `record_outcome` counts the result buckets. Both are no-ops
until setup_telemetry runs, so tests and offline eval are unaffected.
"""
from __future__ import annotations

import os
import time
from typing import Any

_TRACER: Any = None
_M: dict = {}


def setup_telemetry(app) -> None:
    global _TRACER, _M
    from opentelemetry import metrics, trace

    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if conn:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=conn)
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "contract-intel-api"})
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[PrometheusMetricReader()]))
        provider = TracerProvider(resource=resource)
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(provider)

        from fastapi import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics")
        def metrics_endpoint() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")

    meter = metrics.get_meter("contract-intel")
    _TRACER = trace.get_tracer("contract-intel")
    _M = {
        "in_tokens": meter.create_counter("llm.input_tokens"),
        "out_tokens": meter.create_counter("llm.output_tokens"),
        "stage_ms": meter.create_histogram("rag.stage.duration", unit="ms"),
        "outcome": meter.create_counter("answer.outcome"),
        "guardrail": meter.create_counter("guardrail.findings"),
        "requests": meter.create_counter("http.requests"),
        "request_ms": meter.create_histogram("http.request.duration", unit="ms"),
    }

    @app.middleware("http")
    async def _red(request, call_next):  # RED: rate, errors, duration per route
        start = time.perf_counter()
        response = await call_next(request)
        route = request.url.path
        if route not in ("/metrics", "/health"):
            attrs = {"route": route, "method": request.method, "status": str(response.status_code)}
            _M["requests"].add(1, attrs)
            _M["request_ms"].record((time.perf_counter() - start) * 1000, attrs)
        return response


class Traced:
    """Proxy a port to emit a span + stage latency (and LLM tokens) per call."""

    def __init__(self, inner: Any, stage: str) -> None:
        self._inner = inner
        self._stage = stage

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if _TRACER is None or not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            with _TRACER.start_as_current_span(f"rag.{self._stage}"):
                start = time.perf_counter()
                out = attr(*args, **kwargs)
                _M["stage_ms"].record((time.perf_counter() - start) * 1000, {"stage": self._stage})
                if hasattr(out, "input_tokens"):  # LLMResponse
                    _M["in_tokens"].add(out.input_tokens)
                    _M["out_tokens"].add(out.output_tokens)
                return out
        return wrapped


def record_outcome(result: Any) -> None:
    if not _M:
        return
    outcome = ("blocked" if result.blocked else "malformed" if result.malformed
               else "no_context" if result.no_context else "abstained" if result.abstained else "answered")
    _M["outcome"].add(1, {"outcome": outcome})
    for finding in result.findings:
        _M["guardrail"].add(1, {"finding": finding.split(":")[0]})
