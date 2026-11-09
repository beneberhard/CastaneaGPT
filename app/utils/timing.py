"""Per-request latency measurements; never record prompts or credentials."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
from time import perf_counter
import uuid

_current = ContextVar('chat_timing', default=None)


class ChatTiming:
    def __init__(self):
        self.request_id = uuid.uuid4().hex[:12]
        self.started = perf_counter()
        self.seconds = {}
        self.details = {}

    @contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def log(self, status):
        logging.getLogger('uvicorn.error').info('CHAT_TIMING %s', json.dumps({
            'request_id': self.request_id,
            'status': status,
            'total_seconds': round(perf_counter() - self.started, 4),
            'seconds': {k: round(v, 4) for k, v in self.seconds.items()},
            **self.details,
        }))


def record_seconds(name, duration):
    trace = _current.get()
    if trace is not None:
        trace.seconds[name] = trace.seconds.get(name, 0) + duration


def record_detail(name, value):
    trace = _current.get()
    if trace is not None:
        trace.details[name] = value


@contextmanager
def timed_stage(name):
    start = perf_counter()
    try:
        yield
    finally:
        record_seconds(name, perf_counter() - start)
