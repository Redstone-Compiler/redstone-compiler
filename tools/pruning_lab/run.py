#!/usr/bin/env python3
"""Edit lemma expressions below; ground once per variant; prove and benchmark with Z3.
Requires a release-built rsdsl ground example and Python z3-solver.
No changes to the production placer or its default constraints.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
import z3

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl'
TEMPLATE = ROOT / 'docs/rsdsl/inverter_1x3x2.instance.rsdsl'
DEFS = '''
fact incompatible_imp(a: Class, b: Class) := any(on(a, q) and not on(b, q) for q: Case);
fact incompatible_eq(a: Class, b: Class) := any(on(a, q) != on(b, q) for q: Case);
'''
# Each expression is a violation; universal avoidance is the candidate lemma.
BAD = {
    'solid_implication': '''any(Feeds(s, t, f, to) and
        any(Sig[s] is Carry(a) and Sig[t] is Carry(b)
            for a: Class, b: Class where incompatible_imp(a, b))
        for (s, t, f, to) in Feeds where to == SolidSink)''',
    'wire_equivalence': '''any(Feeds(s, t, f, to) and not Cut[s, t] and
        any(Sig[s] is Carry(a) and Sig[t] is Carry(b)
            for a: Class, b: Class where incompatible_eq(a, b))
        for (s, t, f, to) in Feeds where to != SolidSink
            and not (f == FromSolid and to == DustSink))''',
}


def schema_checks(out):
    """Arbitrary single-case Booleans: no dependence on a graph or box size."""
    ctx = z3.Context(proof=True)
    f, cut, a, b, strong = z3.Bools('feeds cut source target strong', ctx=ctx)
    checks = {
        'solid_implication': (z3.Implies(z3.And(f, a), b), z3.And(f, a, z3.Not(b))),
        'wire_equivalence': (z3.Implies(z3.And(f, z3.Not(cut)), a == b),
                             z3.And(f, z3.Not(cut), a != b)),
        # Deliberately unsafe generalizations: the audit must find witnesses.
        'unsafe_ignore_cut': (z3.Implies(z3.And(f, z3.Not(cut)), a == b), z3.And(f, a != b)),
        'unsafe_use_total_block_power': (z3.And(z3.Implies(f, strong == b),
                                                       z3.Implies(strong, a)), z3.And(f, a != b)),
    }
    records = {}
    for name, (premise, violation) in checks.items():
        solver = z3.Solver(ctx=ctx); solver.add(premise, violation)
        result = solver.check()
        records[name] = {'result': str(result)}
        if result == z3.unsat:
            (out / (name + '.z3-proof.txt')).write_text(str(solver.proof()))
        if result == z3.sat:
            records[name]['witness'] = {str(x): str(solver.model().eval(x, model_completion=True))
                                        for x in (f, cut, a, b, strong)}
        expected = z3.sat if name.startswith('unsafe_') else z3.unsat
        assert result == expected, (name, result)
    (out / 'schema.json').write_text(json.dumps(records, indent=2))
    return records


def instance(name, dims, mode, bound=None):
    inputs = ['a'] if mode == 'inv' else ['a', 'b'] if mode == 'nor' else ['s', 'r']
    bits = len(inputs) + (mode == 'rs')
    functions = {n: [bool(q & (1 << i)) for q in range(1 << bits)] for i, n in enumerate(inputs)}
    if mode == 'inv': functions['out'] = [not a for a in functions['a']]
    elif mode == 'nor': functions['out'] = [not (a or b) for a,b in zip(functions['a'],functions['b'])]
    else:
        functions['stored'] = [bool(q & 4) for q in range(8)]
        functions['nq'] = [not (s or q) for s,q in zip(functions['s'],functions['stored'])]
        functions['out'] = [not (r or nq) for r,nq in zip(functions['r'],functions['nq'])]
    for n, vals in list(functions.items()): functions['neg_' + n] = [not v for v in vals]
    functions['off'] = [False] * (1 << bits)
    text = TEMPLATE.read_text()
    replacements = {
        r'instance \w+ of': f'instance {name} of',
        r'grid Cell = .*?;': f'grid Cell = {dims};',
        r'domain Case = .*?;': 'domain Case = [' + ','.join(map(str,range(1 << bits))) + '];',
        r'domain Class = .*?;': 'domain Class = [' + ','.join(functions) + '];',
        r'fact on = .*?;': 'fact on = [' + ','.join(f'({n},{q})' for n,vs in functions.items() for q,v in enumerate(vs) if v) + '];',
        r'fact input_class = .*?;': 'fact input_class = ['+','.join(inputs)+'];',
        r'fact present = .*?;': 'fact present = ['+','.join(inputs)+'];',
        r'fact switch_site = .*?;': 'fact switch_site = [(c, att, k) for c: Cell, att: Attach, k: Class where input_class(k) and has(step(c, dir(att)))];',
        r'fact cut = .*?;': 'fact cut = [(out,stored)];' if mode=='rs' else 'fact cut = [];',
    }
    for pattern,value in replacements.items(): text = re.sub(pattern, value, text)
    # Template comments describe only the reference inverter.
    text = re.sub(r'//[^\n]*', '', text)
    if bound is not None: text += f'\nparam max_blocks = {bound};\n'
    return text


def solve(path, seed, timeout):
    solver = z3.Solver(); solver.set(timeout=timeout, random_seed=seed)
    started = time.perf_counter(); solver.from_file(str(path)); parsed = time.perf_counter()
    status = solver.check(); done = time.perf_counter()
    stats = solver.statistics()
    return {'seed':seed, 'result':str(status), 'load_s':parsed-started, 'solve_s':done-parsed,
            'conflicts':next((stats.get_key_value(k) for k in stats.keys() if k=='sat conflicts'),None)}


def isolated_solve(path, seed, timeout):
    # Fresh Z3 process per run: earlier proofs must not affect solver state.
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             '--solve', str(path), str(seed), str(timeout)],
                            capture_output=True, text=True, timeout=timeout/1000+30)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'target/pruning-lab')
    ap.add_argument('--proof-timeout-ms',type=int,default=30000); ap.add_argument('--seeds',type=int,default=3); ap.add_argument('--timeout-ms',type=int,default=10000)
    args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    source=MODEL.read_text(); records=[]
    schema=schema_checks(out)
    suite=[('inv',(1,3,2),'inv',None),('inv_tight',(1,3,2),'inv',2),
           ('nor',(2,3,3),'nor',None),('rs_latch',(2,3,3),'rs',None)]
    for name,dims,mode,bound in suite:
        ip=out/f'{name}.instance.rsdsl'; ip.write_text(instance(name,dims,mode,bound))
        variants={'base':''}
        for lemma,bad in BAD.items():
            variants['prove_'+lemma]=f'rule "counterexample" {{ require {bad}; }}'
            variants[lemma]=f'rule "lemma" {{ require not ({bad}); }}'
        variants['both']='\n'.join(f'rule "{n}" {{ require not ({v}); }}' for n,v in BAD.items())
        for variant,extra in variants.items():
            mp=out/f'{name}.{variant}.rsdsl'; cp=mp.with_suffix('.cnf')
            mp.write_text(source+DEFS+extra)
            start=time.perf_counter()
            proc=subprocess.run([str(ROOT/'target/release/examples/ground'),str(mp),str(ip),str(cp),'none'],capture_output=True,text=True)
            mp.with_suffix('.ground.log').write_text(proc.stderr)
            if proc.returncode: raise RuntimeError(proc.stderr)
            ground=time.perf_counter()-start
            header=next(l for l in cp.read_text().splitlines() if l.startswith('p cnf'))
            _,_,nv,nc=header.split()
            runs=[isolated_solve(cp,seed,args.proof_timeout_ms if variant.startswith('prove_') else args.timeout_ms) for seed in range(1,2 if variant.startswith('prove_') else args.seeds+1)]
            row={'instance':name,'variant':variant,'variables':int(nv),'clauses':int(nc),'ground_s':ground,'runs':runs}
            records.append(row)
            print(name,variant,nv,nc,[(r['result'],round(r['solve_s'],4)) for r in runs],flush=True)
            (out/'results.json').write_text(json.dumps({'z3':z3.get_version_string(),'model_sha256':hashlib.sha256(source.encode()).hexdigest(),'schema':schema,'records':records},indent=2))
            if variant.startswith('prove_') and any(r['result']=='sat' for r in runs):
                raise RuntimeError(f'Counterexample on {name}: {variant}')
    for name,*_ in suite:
        rows=[r for r in records if r['instance']==name]
        base=next(r for r in rows if r['variant']=='base')
        for row in rows:
            if row['variant'].startswith('prove_'): continue
            assert {r['result'] for r in row['runs'] if r['result']!='unknown'} <= {r['result'] for r in base['runs']} | {'unknown'}
    print('Results:',out/'results.json')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--solve':
        print(json.dumps(solve(Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))))
    else:
        main()
