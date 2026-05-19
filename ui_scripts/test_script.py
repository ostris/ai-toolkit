"""Example UI script invoked via POST /api/scripts.

The API streams stdout/stderr back to the caller. To return structured data,
print a single JSON object on the final line of stdout — the route parses it
into the `result` field of the response.
"""

import argparse
import json
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Example ui_scripts entry point.")
    parser.add_argument("--message", default="hello", help="Message to echo back.")
    parser.add_argument(
        "--count", type=int, default=3, help="Number of log lines to emit."
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Seconds to sleep between log lines."
    )
    parser.add_argument(
        "--fail", action="store_true", help="Exit non-zero to demo failure handling."
    )
    args = parser.parse_args()

    for i in range(args.count):
        print(f"[{i + 1}/{args.count}] {args.message}", flush=True)
        if args.delay > 0:
            time.sleep(args.delay)

    if args.fail:
        print("intentional failure", file=sys.stderr, flush=True)
        return 1

    # Final line: JSON payload the API surfaces as `result`.
    print(
        json.dumps({"message": args.message, "count": args.count, "ok": True}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
