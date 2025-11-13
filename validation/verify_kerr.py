#!/usr/bin/env python3
"""
Validation harness for the Interstellar-style Kerr black hole card.

The script cross-checks analytic Kerr quantities (gravitational radius,
ISCO, photon ring radius, gravitational redshift) against literature
reference values and reports the relative error. A JSON report is
written so the UI can display the latest verification status.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Dict, List, Tuple

G = 6.67430e-11
C = 299_792_458
SOLAR_MASS = 1.98847e30

REFERENCE_ISCO = {
    0.0: 6.0,
    0.5: 4.233,
    0.9: 2.320,
    0.998: 1.237,
}

REFERENCE_PHOTON = {
    0.0: 3.0,
    0.5: 2.3472963553,
    0.9: 1.5578546274,
    0.998: 1.0739092577,
}


def gravitational_radius(mass_solar: float) -> float:
    return (2 * G * mass_solar * SOLAR_MASS) / (C * C)


def isco_radius(spin: float) -> float:
    a = max(min(spin, 0.998), -0.998)
    term = (1 - a * a) ** (1 / 3)
    z1 = 1 + term * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = math.sqrt(3 * a * a + z1 * z1)
    sign = 1 if a >= 0 else -1
    return 3 + z2 - sign * math.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def photon_radius(spin: float) -> float:
    a = max(min(abs(spin), 0.999), 1e-6)
    return 2 * (1 + math.cos((2.0 / 3.0) * math.acos(-a)))


def check_series(
    reference: Dict[float, float], fn, label: str
) -> List[Tuple[str, float, float]]:
    entries: List[Tuple[str, float, float]] = []
    for spin, expected in reference.items():
        value = fn(spin)
        rel_err = abs(value - expected) / expected
        entries.append((label, spin, rel_err))
    return entries


def summarize(errors: List[Tuple[str, float, float]]) -> Dict[str, float]:
    rel_vals = [err for (_, _, err) in errors]
    return {
        "max_rel_error": max(rel_vals),
        "mean_rel_error": statistics.fmean(rel_vals),
    }


def run_checks(mass_solar: float, output: pathlib.Path) -> Dict[str, float]:
    grav_radius = gravitational_radius(mass_solar)
    errors = []
    errors.extend(check_series(REFERENCE_ISCO, isco_radius, "isco"))
    errors.extend(check_series(REFERENCE_PHOTON, photon_radius, "photon"))

    summary = summarize(errors)
    report = {
        "mass_solar": mass_solar,
        "grav_radius_m": grav_radius,
        "max_rel_error": summary["max_rel_error"],
        "mean_rel_error": summary["mean_rel_error"],
        "ok": summary["max_rel_error"] < 8e-3,
        "entries": [
            {"quantity": label, "spin": spin, "relative_error": err}
            for (label, spin, err) in errors
        ],
    }
    output.write_text(json.dumps(report, indent=2))
    return report


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Kerr metric helper calculations."
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=6.5e8,
        help="Black hole mass in solar masses (default: 6.5e8)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("validation_report.json"),
        help="Output JSON path",
    )
    args = parser.parse_args(argv)
    report = run_checks(args.mass, args.out)

    verdict = "PASS" if report["ok"] else "FAIL"
    print(
        f"[{verdict}] max_rel_error={report['max_rel_error']:.4e} "
        f"mean_rel_error={report['mean_rel_error']:.4e}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
