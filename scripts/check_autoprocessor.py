"""Fail with clear error if transformers.AutoProcessor cannot be imported."""
import sys

try:
    from transformers import AutoProcessor
    print("AutoProcessor OK")
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    if getattr(e, "__cause__", None):
        print("CAUSE:", e.__cause__, file=sys.stderr)
    sys.exit(1)
