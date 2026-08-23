import asyncio

import pytest

from capx.serving.drain_batcher import DrainBatcher


@pytest.mark.asyncio
async def test_drains_every_queued_compatible_request() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    batches: list[list[int]] = []

    async def run_batch(kind: str, requests: list[int]) -> list[int]:
        batches.append(requests)
        if len(batches) == 1:
            started.set()
            await release.wait()
        return requests

    batcher = DrainBatcher(run_batch, name="test")
    await batcher.start()
    first = asyncio.create_task(batcher.submit("text", 0))
    await started.wait()
    queued = [asyncio.create_task(batcher.submit("text", value)) for value in range(1, 5)]
    release.set()

    assert await first == 0
    assert await asyncio.gather(*queued) == [1, 2, 3, 4]
    assert batches == [[0], [1, 2, 3, 4]]
    assert batcher.batch_size_counts == {1: 1, 4: 1}
    await batcher.close()


@pytest.mark.asyncio
async def test_oom_guard_leaves_remainder_at_queue_head() -> None:
    batches: list[list[int]] = []

    async def run_batch(kind: str, requests: list[int]) -> list[int]:
        batches.append(requests)
        return requests

    batcher = DrainBatcher(run_batch, max_batch_oom_guard=2, name="test")
    await batcher.start()
    values = await asyncio.gather(*(batcher.submit("text", value) for value in range(5)))

    assert values == list(range(5))
    assert batches == [[0, 1], [2, 3], [4]]
    await batcher.close()


@pytest.mark.asyncio
async def test_one_worker_serializes_incompatible_kinds() -> None:
    batches: list[tuple[str, list[int]]] = []

    async def run_batch(kind: str, requests: list[int]) -> list[int]:
        batches.append((kind, requests))
        return requests

    batcher = DrainBatcher(run_batch, name="test")
    await batcher.start()
    values = await asyncio.gather(
        batcher.submit("text", 1),
        batcher.submit("point", 2),
        batcher.submit("text", 3),
    )

    assert values == [1, 2, 3]
    assert batches == [("text", [1, 3]), ("point", [2])]
    await batcher.close()
