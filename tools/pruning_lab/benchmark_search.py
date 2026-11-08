#!/usr/bin/env python3
"""Interleaved end-to-end ablation; exact layout sets checked after every run."""
import argparse
import hashlib
import json
from pathlib import Path
import z3
from run import ROOT, MODEL
from online_boundary import search, read_problem, self_check, incremental_self_check


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--seconds',type=float,default=30)
    ap.add_argument('--out',type=Path,default=ROOT/'target/search-ablation')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    names=['inv3','inv4','inv5','nor4','nor5','inv_fixed6','inv_fixed7',
           'nor_fixed7','inv_fixed8','inv_long9','nor_fixed10','inv_long10','inv_wide8']
    configs=[('python_smt','allsat','python','smt'),
             ('native_smt','allsat','native','smt'),
             ('packed_smt','allsat_packed','native','smt'),
             ('packed_fd','allsat_packed','native','fd'),
             ('partition_fd','partition','native','fd')]
    result={'z3':z3.get_version_string(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'self_checks':self_check(),'incremental_checks':incremental_self_check(),
            'scope':'all projected physical layouts; CNF generation and reference verification excluded',
            'records':[]}
    for name in names:
        prefix=ROOT/'target/boundary-lab'/name
        fixture=json.loads((ROOT/'tools/pruning_lab/boundary_fixtures'/f'{name}.layouts.json').read_text())
        expected={tuple(x) for x in fixture['layouts']}
        cells,options,_=read_problem(prefix)
        assert cells==list(map(tuple,fixture['cells']))
        assert [[label for label,_ in opts] for opts in options]==fixture['options']
        cut=sum(c[1]<max(c[1] for c in cells) for c in cells)
        for seed in range(1,args.seeds+1):
            offset=(seed-1)%len(configs)
            for label,mode,loader,engine in configs[offset:]+configs[:offset]:
                row,solutions=search(prefix,mode,seed,args.seconds,cut,loader,engine)
                row.update(instance=name,configuration=label,cnf_sha256=hashlib.sha256(prefix.with_suffix('.cnf').read_bytes()).hexdigest())
                if solutions is not None:
                    assert set(solutions)==expected,(name,label,seed)
                    row['exact_set_verified']=True
                else:row['exact_set_verified']=False
                result['records'].append(row)
                print(json.dumps(row),flush=True)
                (args.out/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
