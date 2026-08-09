# Draining vision-service batches

## Change

Each batchable vision endpoint now sends work to one persistent `DrainBatcher`
for its loaded model.  A worker waits for the first request, then drains every
currently queued request of the same inference mode into one model forward.
There is no batching delay and no target batch size.  `max_batch_oom_guard=256`
is the sole limit; overflow remains at the front of the queue for the next
forward.  The worker logs every observed batch size and keeps an in-memory
`batch_size_counts` distribution, rather than treating a configured size as a
performance result.

- SAM3 text uses `Sam3Processor.set_image_batch`, batched text encoding, and
  one `Sam3Image.forward_grounding` call.  Point prompts use SAM3's batch image
  embedding/prompt API in the same state-owning worker.
- OWL-ViT sends image and label lists together to its processor/model, then
  restores each request's threshold and labels during post-processing.
- SAM2 batches box and point prompt mode.  Full-image mask generation remains
  serialized through the same worker because the Transformers pipeline exposes
  no multi-image inference API.

The inventory was confirmed: CaP-X has no DINO server, Molmo is a remote vLLM
client, and Contact-GraspNet, PyRoki, and CuRobo have mutable solver state.
They are intentionally not part of this change.

## Validation performed

`tests/test_drain_batcher.py` verifies queue draining, request order, the OOM
guard remainder, one-worker serialization, and the observed size counter.
`tests/test_sam2_server_batching.py` verifies the box/point batch shapes and
response ordering.  They passed locally (`5 passed`).  `compileall` and
`git diff --check` also pass.

The vendored Meta standalone package was initialized and installed editable
from `capx/third_party/sam3`; this is distinct from Transformers' `sam3_video`
artifact.  Its import advances past package installation after adding its
undeclared runtime dependencies (`einops` and a pre-81 setuptools that still
provides `pkg_resources`), then stops on macOS because Meta SAM3 imports its
CUDA-only `triton` EDT implementation unconditionally.  The local HF cache
also contains `model.safetensors`, not the standalone builder's required
`facebook/sam3` `sam3.pt` checkpoint.  Thus this host cannot execute a real
SAM3 image-model forward, even though MPS is available.

## Ship gate still required on CUDA

No throughput number is claimed.  On a CUDA host with the standalone `sam3.pt`
checkpoint, run the fixed evaluation inputs unbatched twice and concurrently
through the batch service once.  Report:

- unbatched rerun flip rate (the evaluator noise floor),
- batched-vs-unbatched flip rate, which must not exceed that floor (currently
  known as at least 11/315 = 3.5%), and
- the intersection and overlap fraction of those two flip sets.

`scripts/validate_vision_batching.py INPUTS.json --url URL` performs those
three API runs and prints the rates and flip-set overlap.  It is
device-agnostic; use it unchanged against MPS or CUDA, but treat MPS as a
logic-only result.  The service logs must also be collected for
`batch_size_counts`; neither tool reports a configured target batch size or
fabricated throughput.  CUDA results remain required before claiming numerical
equivalence; this macOS result validates only queue and static batching logic.
