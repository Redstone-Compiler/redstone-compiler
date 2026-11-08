#!/usr/bin/env python3
"""Exact projected search lab; defaults to measured-fast native QF_FD AllSAT.
Optional cofactor caches use propagation and bounded Davis-Putnam elimination.
Future Kind literals are protected. No reference solutions are used during search.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import z3
from run import ROOT


def normalize(clauses):
    result=set()
    for clause in clauses:
        c=frozenset(clause)
        if any(-v in c for v in c):continue
        if not c:return ((),)
        result.add(tuple(sorted(c)))
    return tuple(sorted(result))


def propagate(clauses, assumptions, protected):
    values=set(assumptions); clauses=normalize(clauses)
    while True:
        if any(-v in values for v in values):return ((),)
        reduced=[]; units=set()
        for c in clauses:
            if any(v in values for v in c):continue
            r=tuple(v for v in c if -v not in values)
            if not r:return ((),)
            if len(r)==1:units.add(r[0])
            reduced.append(r)
        clauses=tuple(reduced)
        if units<=values:break
        values.update(units)
    # Forget fixed auxiliary values, but retain fixed future decision values.
    return normalize([*clauses,*[(v,) for v in values if abs(v) in protected]])


def summarize(clauses, assumptions, protected, eliminate, indexed=False):
    clauses=propagate(clauses,assumptions,protected)
    if not eliminate or clauses==((),):return clauses
    current=set(map(frozenset,clauses))
    # Deterministic single pass, bounded growth. Skipping a pivot is always safe.
    pivots=sorted({abs(v) for c in current for v in c}-protected)
    occurrences=defaultdict(set)
    if indexed:
        for c in current:
            for literal in c:occurrences[literal].add(c)
    for v in pivots:
        if indexed:pos=list(occurrences[v]);neg=list(occurrences[-v])
        else:pos=[c for c in current if v in c]; neg=[c for c in current if -v in c]
        if not pos and not neg:continue
        if len(pos)*len(neg)>len(pos)+len(neg):continue
        resolvents=set()
        for a in pos:
            for b in neg:
                c=(a-{v})|(b-{-v})
                if not any(-x in c for x in c):resolvents.add(frozenset(c))
        if any(len(c)>12 for c in resolvents):continue
        if indexed:
            for c in pos+neg:
                for literal in c:occurrences[literal].discard(c)
            for c in resolvents:
                for literal in c:occurrences[literal].add(c)
        current.difference_update(pos);current.difference_update(neg);current.update(resolvents)
        if frozenset() in current:return ((),)
    return propagate(current,[],protected)


def holds(clauses,values):return all(any(values[abs(l)]==(l>0) for l in c) for c in clauses)


class IncrementalResidual:
    """Occurrence-indexed unit propagation with reversible clause/assignment edits."""
    def __init__(self, clauses):
        self.original=normalize(clauses)
        self.current=list(self.original)
        self.occurrences=defaultdict(list)
        for i,c in enumerate(self.original):
            for literal in c:self.occurrences[literal].append(i)
        self.values=set();self.trail=[];self.touched=0
        self.conflict=self.original==((),)
        if not self.conflict:self.assign([c[0] for c in self.original if len(c)==1])

    def mark(self):return len(self.trail),self.conflict

    def restore(self, mark):
        length,self.conflict=mark
        while len(self.trail)>length:
            kind,index,old=self.trail.pop()
            if kind=='value':self.values.remove(index)
            else:self.current[index]=old

    def assign(self, pending):
        pending=list(pending)
        while pending and not self.conflict:
            literal=pending.pop()
            if literal in self.values:continue
            if -literal in self.values:self.conflict=True;break
            self.values.add(literal);self.trail.append(('value',literal,None))
            for signed in (literal,-literal):
                for i in self.occurrences[signed]:
                    c=self.current[i]
                    if c is None:continue
                    self.touched+=1
                    updated=None if signed==literal else tuple(v for v in c if v!=signed)
                    self.trail.append(('clause',i,c));self.current[i]=updated
                    if updated==():self.conflict=True;return
                    if updated is not None and len(updated)==1:pending.append(updated[0])

    def key(self, protected):
        if self.conflict:return ((),)
        return tuple(sorted(set(c for c in self.current if c is not None)|
                            {(v,) for v in self.values if abs(v) in protected}))


def incremental_self_check():
    rng=random.Random(23);checks=0
    for _ in range(150):
        clauses=[tuple(rng.choice((-1,1))*rng.randint(1,5) for _ in range(rng.randint(1,4)))
                 for _ in range(rng.randint(1,12))]
        state=IncrementalResidual(clauses)
        def walk(assumptions,depth):
            nonlocal checks
            # Exact canonical comparison also checks protected fixed literals.
            for protected in ({1,2},set(range(1,6)),set()):
                assert state.key(protected)==propagate(clauses,assumptions,protected)
                checks+=1
            if depth==3:return
            for value in (depth+3,-(depth+3)):
                mark=state.mark();state.assign([value])
                walk(assumptions+[value],depth+1);state.restore(mark)
        walk([],0)
    return checks


def self_check():
    rng=random.Random(19);count=0
    for _ in range(150):
        n=5;clauses=[tuple(rng.choice((-1,1))*rng.randint(1,n) for _ in range(rng.randint(1,4))) for _ in range(rng.randint(1,12))]
        protected={1,2};assumptions=[rng.choice((-1,1))*rng.randint(3,5)] if rng.random()<.5 else []
        for eliminate in [False,True]:
            reduced=summarize(clauses,assumptions,protected,eliminate)
            assert reduced==summarize(clauses,assumptions,protected,eliminate,indexed=True)
            for p in itertools.product((False,True),repeat=2):
                possible=[]
                for formula in [clauses+[(v,) for v in assumptions],reduced]:
                    possible.append(any(holds(formula,dict(enumerate(p+a,1))) for a in itertools.product((False,True),repeat=3)))
                assert possible[0]==possible[1],(clauses,assumptions,reduced,p)
                count+=1
    return count


def read_problem(prefix):
    choices=defaultdict(list)
    for line in prefix.with_suffix('.tsv').read_text().splitlines():
        x,y,z,label,lit=line.split('\t');choices[(int(x),int(y),int(z))].append((label,int(lit)))
    clauses=[]
    for line in prefix.with_suffix('.cnf').read_text().splitlines():
        if line and line[0] not in 'pc':clauses.append(tuple(map(int,line.split()[:-1])))
    cells=sorted(choices,key=lambda c:(c[1],c[0],c[2]))
    return cells,[choices[c] for c in cells],clauses


def search(prefix,mode,seed,budget,cut,loader='python',engine='smt'):
    start=time.perf_counter();cells,options,clauses=read_problem(prefix)
    ids={abs(v) for opts in options for _,v in opts} if loader=='native' else {abs(v) for c in clauses for v in c}
    # DIMACS uses integer symbols: Bool(1) is distinct from Bool('k!1').
    vars={v:z3.Bool(v if loader=='native' else f'v{v}') for v in ids}
    literals={v:b for v,b in vars.items()}
    literals.update({-v:z3.Not(b) for v,b in vars.items()})
    def lit(v):return literals[v]
    solver=z3.Tactic('sat').solver() if engine=='sat' else z3.SolverFor('QF_FD') if engine=='fd' else z3.Solver()
    solver.set(timeout=5000,random_seed=seed)
    if loader=='native':solver.from_file(str(prefix.with_suffix('.cnf')))
    else:solver.add([z3.Or([lit(v) for v in c]) for c in clauses])
    packed=None
    if mode in {'allsat_packed','partition'}:
        radix=max(map(len,options))
        packed=z3.Sum([z3.If(lit(v),i*radix**j,0)
                       for j,opts in enumerate(options) for i,(_,v) in enumerate(opts) if i])
    def decode(model):
        if packed is None:
            return tuple(next(i for i,(_,v) in enumerate(opts) if z3.is_true(model.eval(lit(v),model_completion=True))) for opts in options)
        value=model.eval(packed,model_completion=True).as_long();selected=[]
        for opts in options:
            value,i=divmod(value,radix);assert i<len(opts);selected.append(i)
        assert value==0
        return tuple(selected)
    guided=mode in {'forward','forward_cache','lazy','lazy_cache'}
    state=IncrementalResidual(clauses) if mode.startswith('incremental') or guided else None
    setup=time.perf_counter()-start
    protected={abs(v) for opts in options[cut:] for _,v in opts}
    cache={};stats={'checks':0,'cache_hits':0,'summaries':0,'summary_s':0.0,'update_s':0.0,'reused_suffix_layouts':0,
                    'propagation_conflicts':0,'excluded_options':0,'entailed_options':0}
    def guided_dfs(depth,assumptions,known_sat=False):
        if time.perf_counter()-start>budget:raise TimeoutError('wall budget')
        if state.conflict:
            stats['propagation_conflicts']+=1;return ()
        key=None
        if mode.endswith('_cache') and depth==cut:
            before=time.perf_counter();key=state.key(protected)
            stats['summary_s']+=time.perf_counter()-before;stats['summaries']+=1
            if key in cache:
                stats['cache_hits']+=1;stats['reused_suffix_layouts']+=len(cache[key]);return cache[key]
        if not known_sat and (mode.startswith('forward') or depth==len(options)):
            stats['checks']+=1
            status=solver.check(*[lit(v) for v in assumptions])
            if status==z3.unknown:raise TimeoutError(solver.reason_unknown())
            if status==z3.unsat:
                if key is not None:cache[key]=()
                return ()
            known_sat=True
        if depth==len(options):return ((),)
        suffixes=[]
        for i,(_,v) in enumerate(options[depth]):
            if -v in state.values:
                stats['excluded_options']+=1;continue
            if v in state.values:
                stats['entailed_options']+=1
                tails=guided_dfs(depth+1,assumptions,known_sat)
            else:
                before=time.perf_counter();mark=state.mark();state.assign([v])
                stats['update_s']+=time.perf_counter()-before
                try:tails=guided_dfs(depth+1,assumptions+[v])
                finally:
                    before=time.perf_counter();state.restore(mark)
                    stats['update_s']+=time.perf_counter()-before
            suffixes.extend((i,)+tail for tail in tails)
        result=tuple(suffixes)
        if key is not None:cache[key]=result
        return result
    def dfs(depth,assumptions):
        if time.perf_counter()-start>budget:raise TimeoutError('wall budget')
        stats['checks']+=1
        status=solver.check(*[lit(v) for v in assumptions])
        if status==z3.unknown:raise TimeoutError(solver.reason_unknown())
        if status==z3.unsat:return ()
        mark=None
        if state is not None and 0<depth<=cut:
            before=time.perf_counter();mark=state.mark();state.assign([assumptions[-1]])
            stats['update_s']+=time.perf_counter()-before
            assert not state.conflict
        try:return expand(depth,assumptions)
        finally:
            if mark is not None:
                before=time.perf_counter();state.restore(mark)
                stats['update_s']+=time.perf_counter()-before

    def expand(depth,assumptions):
        if depth==len(options):return ((),)
        key=None
        if mode!='baseline' and depth==cut:
            before=time.perf_counter()
            if state is not None:
                key=state.key(protected)
                if mode=='incremental_eliminate':key=summarize(key,[],protected,True,indexed=True)
            else:key=summarize(clauses,assumptions,protected,mode=='eliminate')
            stats['summary_s']+=time.perf_counter()-before;stats['summaries']+=1
            if key in cache:
                stats['cache_hits']+=1;stats['reused_suffix_layouts']+=len(cache[key]);return cache[key]
        suffixes=[]
        for i,(_,v) in enumerate(options[depth]):
            suffixes.extend((i,)+tail for tail in dfs(depth+1,assumptions+[v]))
        result=tuple(suffixes)
        if key is not None:cache[key]=result
        return result
    def allsat():
        solutions=[]
        while True:
            if time.perf_counter()-start>budget:raise TimeoutError('wall budget')
            stats['checks']+=1
            status=solver.check()
            if status==z3.unknown:raise TimeoutError(solver.reason_unknown())
            if status==z3.unsat:return tuple(solutions)
            model=solver.model()
            selected=decode(model)
            solutions.append(selected)
            solver.add(z3.Or([z3.Not(lit(opts[i][1])) for opts,i in zip(options,selected)]))
    def partition(depth,witness=None):
        # Partition the physical domain; one SAT witness serves its entire path.
        if time.perf_counter()-start>budget:raise TimeoutError('wall budget')
        if witness is None:
            stats['checks']+=1;status=solver.check()
            if status==z3.unknown:raise TimeoutError(solver.reason_unknown())
            if status==z3.unsat:return ()
            witness=decode(solver.model())
        if depth==len(options):return ((),)
        result=[]
        solver.push()
        try:
            while True:
                i=witness[depth];v=options[depth][i][1]
                solver.push();solver.add(lit(v))
                try:tails=partition(depth+1,witness)
                finally:solver.pop()
                result.extend((i,)+tail for tail in tails)
                solver.add(z3.Not(lit(v)))
                stats['checks']+=1;status=solver.check()
                if status==z3.unknown:raise TimeoutError(solver.reason_unknown())
                if status==z3.unsat:break
                witness=decode(solver.model())
        finally:solver.pop()
        return tuple(result)
    try:
        solutions=allsat() if mode.startswith('allsat') else partition(0) if mode=='partition' else guided_dfs(0,[]) if guided else dfs(0,[]);status='complete'
    except TimeoutError as e:
        solutions=None;status='incomplete: '+str(e)
    elapsed=time.perf_counter()-start
    record={'mode':mode,'loader':loader,'engine':engine,'seed':seed,'cut_variables':cut,'status':status,'total_s':elapsed,'setup_s':setup,
            **stats,'updated_clause_visits':state.touched if state else 0,
            'cache_entries':len(cache),'cached_clause_literals':sum(sum(map(len,k)) for k in cache)}
    if solutions is not None:
        canonical=sorted(set(solutions));assert len(canonical)==len(solutions)
        record.update(model_count=len(canonical),layout_sha256=hashlib.sha256(json.dumps(canonical).encode()).hexdigest())
    return record,solutions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--instances',default='inv3,inv4,inv5,nor4,nor5')
    ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--seconds',type=float,default=60)
    ap.add_argument('--modes',default='allsat_packed')
    ap.add_argument('--loader',choices=['python','native'],default='native')
    ap.add_argument('--engine',choices=['smt','sat','fd'],default='fd')
    ap.add_argument('--out',type=Path,default=ROOT/'target/online-boundary');args=ap.parse_args()
    allowed={'baseline','propagate','eliminate','incremental','incremental_eliminate','allsat',
             'forward','forward_cache','lazy','lazy_cache','allsat_packed','partition'}
    if not set(args.modes.split(','))<=allowed:ap.error('unknown search mode')
    args.out.mkdir(parents=True,exist_ok=True)
    records=[];checks=self_check();incremental_checks=incremental_self_check()
    for name in args.instances.split(','):
        prefix=ROOT/'target/boundary-lab'/name
        reference=json.loads((ROOT/'tools/pruning_lab/boundary_fixtures'/f'{name}.layouts.json').read_text())
        expected={tuple(x) for x in reference['layouts']}
        cells,_,_=read_problem(prefix)
        # Leave one complete spatial column to the right (best observed tiny compression).
        cut=sum(c[1]<max(c[1] for c in cells) for c in cells)
        for seed in range(1,args.seeds+1):
            modes=args.modes.split(',');offset=(seed-1)%len(modes);modes=modes[offset:]+modes[:offset]
            for mode in modes:
                row,solutions=search(prefix,mode,seed,args.seconds,cut,args.loader,args.engine);row['instance']=name
                if solutions is not None:assert set(solutions)==expected,(name,mode)
                records.append(row);print(json.dumps(row),flush=True)
                (args.out/'results.json').write_text(json.dumps({'self_check_cases':checks,'incremental_check_cases':incremental_checks,'z3':z3.get_version_string(),
                        'definition':'exact projected search; reference fixtures used only after search for validation',
                        'records':records},indent=2))
if __name__=='__main__':main()
