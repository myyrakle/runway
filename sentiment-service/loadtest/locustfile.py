"""Locust load test for the sentiment service.

Fires batched POST /invocations requests built from a sample chunk file shaped
like the endpoint's payload: {"aspect": "...", "texts": [...]}. Each request
sends BATCH_SIZE texts sampled from the file; concurrency is the number of
Locust users (closed-loop: a user fires the next request as soon as the previous
response returns, so N users ≈ N in-flight requests).

Install:
  uv pip install locust            # or: uv sync --extra loadtest

Run — 100 concurrent users, web UI at http://localhost:8089:
  uv run locust -f loadtest/locustfile.py --host http://localhost:8001 -u 100 -r 100

Run — headless, 100 users, 2 minutes, CSV report:
  uv run locust -f loadtest/locustfile.py --host http://localhost:8001 \
    -u 100 -r 100 -t 2m --headless --csv loadtest/report

Knobs (env vars):
  SAMPLE_FILE  path to the {aspect, texts} JSON   (default: /home/myyrakle/Download/chunk-0.json)
  BATCH_SIZE   texts per request                  (default: 100; capped at file size)
  ASPECT       override the aspect                (default: from file)
  ENDPOINT     endpoint path                       (default: /invocations)
  PRECISION    precision variant to request        (default: unset -> server default)

`-u` = concurrent users (the "동시에"), BATCH_SIZE = texts per request (the "100개씩").
"""
from __future__ import annotations

import json
import os
import random

from locust import HttpUser, constant, task

SAMPLE_FILE = os.environ.get("SAMPLE_FILE", "/home/myyrakle/Download/chunk-0.json")
ENDPOINT = os.environ.get("ENDPOINT", "/invocations")

with open(SAMPLE_FILE, encoding="utf-8") as _f:
    _DATA = json.load(_f)

_TEXTS: list[str] = _DATA["texts"]
if not _TEXTS:
    raise RuntimeError(f"{SAMPLE_FILE} has no texts")

_ASPECT = os.environ.get("ASPECT", _DATA.get("aspect", "overall"))
_PRECISION = os.environ.get("PRECISION") or None
_BATCH = min(int(os.environ.get("BATCH_SIZE", "100")), len(_TEXTS))


class SentimentUser(HttpUser):
    # Closed-loop: no think time, so each user keeps one request in flight and the
    # offered concurrency equals the user count. Switch to between(...) for open-loop.
    wait_time = constant(0)

    @task
    def invocations(self) -> None:
        # Sample per request so payloads vary in order/content rather than hammering
        # one identical batch. With BATCH_SIZE == file size this just shuffles.
        texts = _TEXTS if _BATCH >= len(_TEXTS) else random.sample(_TEXTS, _BATCH)
        payload: dict = {"texts": texts, "aspect": _ASPECT}
        if _PRECISION:
            payload["precision"] = _PRECISION

        with self.client.post(
            ENDPOINT, json=payload, catch_response=True, name=f"{ENDPOINT} (batch={_BATCH})"
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                results = resp.json().get("results", [])
            except ValueError:
                resp.failure("response is not JSON")
                return
            if len(results) != len(texts):
                resp.failure(f"expected {len(texts)} results, got {len(results)}")
