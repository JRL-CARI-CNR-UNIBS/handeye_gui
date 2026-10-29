# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Offline recomputation from a saved session folder.

Examples::

    handeye_cli ~/calib/session_01
    handeye_cli ~/calib/session_01 --method all --ratio 0.7 --bootstrap 200
    handeye_cli ~/calib/session_01 --exclude 3 7 --write
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .core.evaluation import evaluate, format_report
from .core.geometry import pose7_to_matrix
from .core.session import RESULTS_FILE, Session
from .core.solvers import Method


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="handeye_cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="session folder (config.yaml, samples.yaml, images/)")
    p.add_argument("--method", default="TSAI",
                   help="TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS or 'all' (default TSAI)")
    p.add_argument("--ratio", type=float, default=0.7,
                   help="identification fraction, 1.0 = no validation (default 0.7)")
    p.add_argument("--seed", type=int, default=0, help="random seed for split and bootstrap")
    p.add_argument("--bootstrap", type=int, default=0, metavar="N", help="bootstrap rounds")
    p.add_argument("--sigma", type=float, default=None,
                   help="outlier threshold in std (default: config outlier_sigma)")
    p.add_argument("--exclude", type=int, nargs="*", default=None, metavar="IDX",
                   help="exclude these sample indices (replaces the flags saved in the session)")
    p.add_argument("--write", action="store_true",
                   help=f"write {RESULTS_FILE} (and the exclusion flags) back to the folder")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    session = Session.load(args.folder)
    if args.exclude is not None:
        for s in session.samples:
            s.excluded = s.index in set(args.exclude)
    n_sigma = args.sigma if args.sigma is not None else session.config.outlier_sigma
    T_oc = session.metadata.get("T_optical_camera")
    T_oc = pose7_to_matrix(T_oc) if T_oc else None
    distance_range = (session.config.random_z_min, session.config.random_z_max)
    methods = list(Method) if args.method.lower() == "all" else [Method.parse(args.method)]

    print(f"Session {args.folder}: {len(session.samples)} samples, "
          f"{len(session.active_samples())} used")
    reports = {}
    for m in methods:
        try:
            rep = evaluate(session.samples, m, args.ratio, args.seed, args.bootstrap, n_sigma,
                           T_oc, distance_range)
        except (ValueError, RuntimeError) as exc:
            print(f"\n== {m.name}: failed: {exc}")
            continue
        reports[m] = rep
        print(f"\n== {m.name}")
        print(format_report(rep))
    if not reports:
        return 1
    if len(reports) > 1:
        print("\nSummary (validation translation error, mm):")
        for m, rep in reports.items():
            v = rep.valid_stats or rep.ident_stats
            print(f"  {m.name:<11} {1e3 * v.translation_mean:8.3f} +/- {1e3 * v.translation_std:.3f}"
                  f"  angle {v.angle_mean:.3f} deg")
    if args.write:
        best = min(reports.values(),
                   key=lambda r: (r.valid_stats or r.ident_stats).translation_mean)
        session.results = best.to_dict()
        session.save(args.folder)
        print(f"\nWrote {RESULTS_FILE} ({best.result.method.name})")
    return 0


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    sys.exit(main())
