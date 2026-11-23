"""Run the isolated v2 experiment phases."""
from __future__ import annotations

import argparse
import sys

from src import config, experiment


def main():
    parser = argparse.ArgumentParser(description="CS5185 v2 adaptive video encoding")
    commands = parser.add_subparsers(dest="command", required=True)
    features = commands.add_parser("features", help="extract source features; includes TEST only for reporting")
    features.add_argument("--run", default="features_v2")
    sweep = commands.add_parser("sweep", help="scan pre-registered candidates on DEV only")
    sweep.add_argument("--run", required=True)
    derive = commands.add_parser("derive", help="derive DEV-only rules and run LOOCV")
    derive.add_argument("--run", required=True, help="sweep run containing sweep_summary.csv")
    validate = commands.add_parser("validate", help="check frozen rules on VAL only")
    validate.add_argument("--run", required=True)
    final = commands.add_parser("final", help="run the one-shot TEST pass; needs a completed VAL run with unchanged rules/code")
    final.add_argument("--run", required=True)
    final.add_argument("--val", required=True, help="VAL run that froze the rules")
    align = commands.add_parser("align-check", help="compare custom MS-SSIM against pytorch-msssim")
    align.add_argument("--run", required=True, help="completed DEV sweep run")
    checks = commands.add_parser("check-retry", help="trigger the isolated E3 over-target retry")
    checks.add_argument("--run", required=True, help="new directory name under results/checks/")
    everything = commands.add_parser("all", help="run the whole pipeline from features to report")
    everything.add_argument("--prefix", required=True, help="stages are written to <prefix>_features/_sweep/_val/_final")
    report = commands.add_parser("report", help="build figures and TEST subjective materials")
    report.add_argument("--run", required=True, help="final run to build the report for")
    args = parser.parse_args()

    try:
        if args.command == "features":
            path = experiment.run_features(args.run)
        elif args.command == "sweep":
            path = experiment.sweep(args.run)
        elif args.command == "derive":
            result = experiment.derive(args.run)
            print(f"Rules written: {result['path']} ({len(result['rules'])} rules)")
            path = config.RUN_DIR / args.run
        elif args.command == "validate":
            path = experiment.validate(args.run)
        elif args.command == "final":
            path = experiment.final(args.run, args.val)
        elif args.command == "align-check":
            path = experiment.alignment_check(args.run)
        elif args.command == "check-retry":
            path = experiment.over_target_check(args.run)
        elif args.command == "all":
            path = experiment.run_all(args.prefix)
        else:
            from src import report as report_module
            path = report_module.generate(args.run)
        print(f"Run output: {path}")
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
