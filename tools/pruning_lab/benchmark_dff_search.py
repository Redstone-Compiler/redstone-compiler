#!/usr/bin/env python3
"""Serial, cold, release-mode DFF search comparisons; never loads a placement."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--options", default="")
parser.add_argument("--model", default="docs/rsdsl/dff_global_fast.rsdsl")
parser.add_argument("--params", default="")
parser.add_argument("--label", required=True)
parser.add_argument("--seeds", default="11")
parser.add_argument("--seconds", type=int, default=180)
args = parser.parse_args()
out = ROOT / "target" / "dff-search-bench" / args.label
out.mkdir(parents=True, exist_ok=True)
rows = []
for seed in map(int, args.seeds.split(",")):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEQ_")}
    env.update(EXACT_PROFILE="1", SEQ_CIRCUIT="dff-neg", SEQ_DIMS="2x7x4",
               SEQ_SECONDS=str(args.seconds), SEQ_WORKERS="4", SEQ_OPTIMIZE="0",
               SEQ_RANK="8", SEQ_STAGE="8", SEQ_SYM="off", SEQ_SEED=str(seed),
               SEQ_PRUNE="0", SEQ_SOLVER_OPTIONS=args.options,
               SEQ_MODEL_FILE=args.model, SEQ_MODEL_PARAMS=args.params,
               SEQ_WRITE=str(out / str(seed)))
    log_path = out / f"{seed}.log"
    with log_path.open("w") as log:
        proc = subprocess.run(["cargo", "test", "--release", "--lib",
                               "explore_sequential_cells", "--", "--ignored", "--nocapture"],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = log_path.read_text()
    found = re.search(r"SEQ placed .*?: (\d+) blocks.*?discovery ([\d.]+)s", log)
    stats = re.search(r"SEQ stats vars=(\d+) clauses=(\d+).*?solve=([^ ]+).*?refinements=(\d+)", log)
    row = dict(seed=seed, options=args.options, model=args.model, params=args.params, exit_code=proc.returncode,
               placed=bool(found), limit_s=args.seconds, log=str(log_path.relative_to(ROOT)))
    if found:
        row.update(blocks=int(found[1]), discovery_s=float(found[2]))
    if stats:
        row.update(variables=int(stats[1]), clauses=int(stats[2]),
                   solve=stats[3], refinements=int(stats[4]))
    rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
