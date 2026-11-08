#!/usr/bin/env python3
"""Summarize paired rank-normalization cold searches, retaining unknown runs."""
import json
import statistics
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
summary = []
for name in dict.fromkeys(row["case"] for row in rows):
    group = [row for row in rows if row["case"] == name]
    record = {"case": name}
    for enabled, label in [(False, "off"), (True, "on")]:
        sample = [row for row in group if row["enabled"] == enabled]
        if not sample:
            continue
        solved = [row for row in sample if row["status"] == "placed"]
        record[label] = {
            "runs": len(sample),
            "solved": len(solved),
            "unknown": sum(row["status"] == "unknown" for row in sample),
            "infeasible": sum(row["status"] == "infeasible" for row in sample),
            "median_wall_s": statistics.median(row["wall_s"] for row in sample),
            "median_solved_wall_s": statistics.median(row["wall_s"] for row in solved) if solved else None,
            "median_encode_s": statistics.median(row["encode_s"] for row in sample),
            "median_solve_s": statistics.median(row["solve_s"] for row in sample),
            # Failure must not look faster just because a worker gives up early.
            "mean_par2_s": statistics.mean(row["wall_s"] if row["status"] == "placed"
                                           else row["encode_s"] + 2 * row["limit_s"]
                                           for row in sample),
        }
    seeds = sorted(set(row["seed"] for row in group))
    pairs = []
    for seed in seeds:
        pair = {row["enabled"]: row for row in group if row["seed"] == seed}
        if len(pair) != 2:
            continue
        off, on = pair[False], pair[True]
        both_solved = off["status"] == on["status"] == "placed"
        pairs.append({"seed": seed, "off_status": off["status"], "on_status": on["status"],
                      "off_wall_s": off["wall_s"], "on_wall_s": on["wall_s"],
                      "speedup_if_both_solved": off["wall_s"] / on["wall_s"] if both_solved else None})
    record["pairs"] = pairs
    summary.append(record)
print(json.dumps(summary, indent=2))
