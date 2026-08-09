"""A single-worker request queue that drains compatible work into GPU batches."""

import asyncio
import inspect
import logging
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass
class _QueuedRequest(Generic[RequestT, ResultT]):
    kind: str
    request: RequestT
    future: asyncio.Future[ResultT]


class DrainBatcher(Generic[RequestT, ResultT]):
    """Run one compatible queue drain at a time.

    There is deliberately no batching timeout or target batch size.  After a
    worker finishes, it takes every compatible request already waiting, up to
    ``max_batch_oom_guard``.  Requests of another kind stay queued for the
    same worker, which keeps one model's mutable inference state serialized.
    """

    def __init__(
        self,
        run_batch: Callable[
            [str, Sequence[RequestT]], Sequence[ResultT] | Awaitable[Sequence[ResultT]]
        ],
        *,
        max_batch_oom_guard: int = 256,
        name: str = "vision",
    ) -> None:
        if max_batch_oom_guard < 1:
            raise ValueError("max_batch_oom_guard must be positive")
        self._run_batch = run_batch
        self._max_batch_oom_guard = max_batch_oom_guard
        self._name = name
        self._pending: deque[_QueuedRequest[RequestT, ResultT]] = deque()
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self.batch_size_counts: Counter[int] = Counter()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._serve(), name=f"{self._name}-batcher")

    async def close(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        with suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None

    async def submit(self, kind: str, request: RequestT) -> ResultT:
        if self._worker is None:
            raise RuntimeError(f"{self._name} batcher has not been started")
        future: asyncio.Future[ResultT] = asyncio.get_running_loop().create_future()
        async with self._condition:
            self._pending.append(_QueuedRequest(kind, request, future))
            self._condition.notify()
        return await future

    async def _next_batch(self) -> list[_QueuedRequest[RequestT, ResultT]]:
        async with self._condition:
            while not self._pending:
                await self._condition.wait()

            kind = self._pending[0].kind
            batch: list[_QueuedRequest[RequestT, ResultT]] = []
            remainder: deque[_QueuedRequest[RequestT, ResultT]] = deque()
            while self._pending:
                item = self._pending.popleft()
                if item.kind == kind and len(batch) < self._max_batch_oom_guard:
                    batch.append(item)
                else:
                    remainder.append(item)
            self._pending = remainder
            return batch

    async def _serve(self) -> None:
        while True:
            batch = await self._next_batch()
            kind = batch[0].kind
            self.batch_size_counts[len(batch)] += 1
            logger.info("%s running %s batch of %d", self._name, kind, len(batch))
            try:
                if inspect.iscoroutinefunction(self._run_batch):
                    results = await self._run_batch(kind, [item.request for item in batch])
                else:
                    results = await asyncio.to_thread(
                        self._run_batch, kind, [item.request for item in batch]
                    )
                    if inspect.isawaitable(results):
                        results = await results
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"{self._name} {kind} batch returned {len(results)} results for {len(batch)} requests"
                    )
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            else:
                for item, result in zip(batch, results, strict=True):
                    if not item.future.done():
                        item.future.set_result(result)
