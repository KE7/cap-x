#!/usr/bin/env python3
"""Compare sequential and concurrent vision-service responses on fixed inputs.

The input JSON is a list of request bodies accepted by one endpoint.  Running
with one request at a time exercises the service's lone-request batch path;
running concurrently lets its drain worker form emergent batches.  This tool is
device-agnostic and deliberately reports no throughput number.
"""

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _post(url: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - user-provided local URL
            return json.load(response)
    except urllib.error.HTTPError as exc:
        return {"http_error": exc.code, "body": exc.read().decode(errors="replace")}


def _run(url: str, payloads: list[dict[str, Any]], concurrent: bool) -> list[Any]:
    if not concurrent:
        return [_post(url, payload) for payload in payloads]
    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        return list(executor.map(lambda payload: _post(url, payload), payloads))


def _flip_set(reference: list[Any], candidate: list[Any]) -> set[int]:
    return {
        index
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True))
        if left != right
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, help="JSON array of endpoint request bodies")
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8114/segment")
    args = parser.parse_args()
    payloads = json.loads(args.inputs.read_text())
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("inputs must be a non-empty JSON array")

    unbatched_first = _run(args.url, payloads, concurrent=False)
    unbatched_second = _run(args.url, payloads, concurrent=False)
    batched = _run(args.url, payloads, concurrent=True)
    noise_flips = _flip_set(unbatched_first, unbatched_second)
    batch_flips = _flip_set(unbatched_first, batched)
    overlap = noise_flips & batch_flips
    total = len(payloads)
    print(
        json.dumps(
            {
                "inputs": total,
                "unbatched_rerun": {"flips": len(noise_flips), "rate": len(noise_flips) / total},
                "batched_vs_unbatched": {"flips": len(batch_flips), "rate": len(batch_flips) / total},
                "flip_overlap": {
                    "count": len(overlap),
                    "of_batched_flips": len(overlap) / len(batch_flips) if batch_flips else 1.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
