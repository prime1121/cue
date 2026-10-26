"""Run a Python suite and reject skipped or empty verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"python": ROOT / "python/tests", "models": ROOT / "scripts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / "python/src"), str(SUITES[args.suite])]
    tests = unittest.defaultTestLoader.discover(str(SUITES[args.suite]), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(tests)
    passed = bool(result.testsRun) and result.wasSuccessful() and not result.skipped
    print(json.dumps({"suite": args.suite, "tests": result.testsRun,
                      "failures": len(result.failures), "errors": len(result.errors),
                      "skipped": len(result.skipped), "passed": bool(passed)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
