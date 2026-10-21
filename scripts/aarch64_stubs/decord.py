"""Stub for `decord` (no aarch64 wheel). SAM3 only uses decord for VIDEO frame
reading (sam3_image_dataset / sam2_utils video path), which the IMAGE-inference
server (launch_sam3_server.py: set_image/set_text_prompt/predict_inst) never hits.
Importing this stub satisfies the eager `from decord import cpu, VideoReader`
in train/data modules pulled onto the import path. Any actual use raises clearly."""
def _unavailable(*a, **k):
    raise RuntimeError("decord is stubbed on aarch64 (video path unused for image inference)")
class VideoReader:  # noqa: N801
    def __init__(self, *a, **k):
        _unavailable()
def cpu(*a, **k):
    return None
def gpu(*a, **k):
    return None
class bridge:  # noqa: N801
    @staticmethod
    def set_bridge(*a, **k):
        return None
