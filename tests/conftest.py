from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest

from slogate.fake_engine import EngineProfile, FakeEngine, serve_in_thread


@pytest.fixture
def fake_engine() -> Iterator[Callable[..., str]]:
    """Start fake engines on free ports; every engine stops when the test ends."""
    with ExitStack() as stack:

        def start(**profile: object) -> str:
            engine = FakeEngine(EngineProfile(**profile))  # type: ignore[arg-type]
            return stack.enter_context(serve_in_thread(engine))

        yield start
