"""Run several pipeline commands in order inside one background job.
  python -m pipeline.batch "stage6 retry shot_a" "stage6 retry shot_b" ...
"""
import importlib
import sys
import time

for spec in sys.argv[1:]:
    mod, *args = spec.split()
    print(f"[{time.strftime('%H:%M:%S')}] batch: {spec}", flush=True)
    try:
        importlib.import_module(f"pipeline.{mod}").main(args)
    except SystemExit:
        pass
    except Exception as e:  # keep going; failures are visible in the log and the manifest
        print(f"[{time.strftime('%H:%M:%S')}] batch error in {spec}: {e!r}", flush=True)
