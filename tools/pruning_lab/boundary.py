#!/usr/bin/env python3
"""Exact projected-model enumeration for tiny spatial continuation experiments.
A boundary summary is safe iff equal summaries have identical right-completion sets.
This is finite-instance evidence for the static rsdsl model, not Minecraft completeness.
"""
import argparse
from collections import defaultdict
import itertools
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import z3
from run import ROOT, MODEL, instance


def load(prefix):
    choices = defaultdict(list)
    for line in prefix.with_suffix('.tsv').read_text().splitlines():
        x,y,z,label,lit=line.split('\t')
        choices[(int(x),int(y),int(z))].append((label,int(lit)))
    cells=sorted(choices)
    lines=prefix.with_suffix('.cnf').read_text().splitlines()
    n=int(next(l for l in lines if l.startswith('p ')).split()[2])
    variables=[None]+[z3.Bool(f'v{i}') for i in range(1,n+1)]
    def literal(n): return variables[n] if n>0 else z3.Not(variables[-n])
    solver=z3.Solver(); solver.set(timeout=5000)
    for l in lines:
        if not l or l[0] in 'pc': continue
        solver.add(z3.Or([literal(int(x)) for x in l.split()[:-1]]))
    options=[[ (label,literal(lit)) for label,lit in choices[c] ] for c in cells]
    return solver,cells,options


def enumerate_layouts(solver,options,limit,seconds):
    start=time.perf_counter(); layouts=[]; reason='limit'
    solver.push()
    while len(layouts)<limit and time.perf_counter()-start<seconds:
        result=solver.check()
        if result==z3.unsat: reason='exhausted'; break
        if result!=z3.sat: reason='unknown'; break
        model=solver.model(); selected=[]; lits=[]
        for choices in options:
            matches=[(i,expr) for i,(_,expr) in enumerate(choices) if z3.is_true(model.eval(expr,model_completion=True))]
            assert len(matches)==1, matches
            i,expr=matches[0]; selected.append(i); lits.append(expr)
        layouts.append(tuple(selected)); solver.add(z3.Not(z3.And(lits)))
        if len(layouts)%100==0: print('  layouts',len(layouts),flush=True)
    solver.pop()
    return layouts,reason,time.perf_counter()-start


def analyze(solver,cells,options,layouts,cut,budget):
    left=[i for i,c in enumerate(cells) if c[1]<cut]
    right=[i for i,c in enumerate(cells) if c[1]>=cut]
    completions=defaultdict(set)
    for row in layouts:
        completions[tuple(row[i] for i in left)].add(tuple(row[i] for i in right))
    domains=[range(len(options[i])) for i in left]
    universe_size=math.prod(map(len,domains))
    if universe_size>2_000_000:
        return {'cut':cut,'status':'prefix_universe_too_large','prefix_cartesian_size':universe_size}
    names=('empty','resources','face_types','face_exact','face_resources','halo2_resources')
    groups={n:{} for n in names}; livegroups={n:{} for n in names}; collisions={n:None for n in names}
    prefixes=0; local_prefixes=0; residuals=set(); live_residuals=set(); cost_residuals=set()
    # Includes dead prefixes; optional filters are left-only necessary conditions.
    for key in itertools.product(*domains):
        labels=[options[i][v][0] for i,v in zip(left,key)]
        occupied=sum(v!='Air' for v in labels)
        if occupied>budget: continue
        switches=tuple(sorted(v for v in labels if v.startswith('Switch(')))
        input_names=[v.rsplit(',',1)[1][:-1] for v in switches]
        if len(input_names)!=len(set(input_names)): continue
        prefixes+=1
        signature=frozenset(completions.get(key,()))
        residuals.add(signature)
        if signature:
            local_prefixes+=1; live_residuals.add(signature); cost_residuals.add((signature,occupied))
        face=tuple(v for i,v in zip(left,labels) if cells[i][1]==cut-1)
        halo=tuple(v for i,v in zip(left,labels) if cells[i][1]>=cut-2)
        resources=(occupied,tuple(sorted(input_names)))
        summaries=((), resources, tuple(v.split('(')[0] for v in face),face,(face,resources),(halo,resources))
        for name,summary in zip(names,summaries):
            previous=groups[name].get(summary)
            if previous is None: groups[name][summary]=(signature,key)
            elif previous[0]!=signature and collisions[name] is None:
                collisions[name]=(previous[1],key,previous[0],signature)
            if signature:
                prior=livegroups[name].get(summary)
                if prior is None: livegroups[name][summary]=(signature,key)
                elif prior[0]!=signature:
                    # Prefer a witness where both left prefixes can be extended.
                    existing=collisions[name]
                    if existing is None or not existing[2] or not existing[3]:
                        collisions[name]=(prior[1],key,prior[0],signature)
    reports=[]
    for name in names:
        witness=None; hit=collisions[name]
        if hit:
            a,b,sa,sb=hit
            if not sa-sb: a,b,sa,sb=b,a,sb,sa
            suffix=next(iter(sa-sb))
            checks=[]
            for key in (a,b):
                constraints=[options[i][v][1] for i,v in zip(left,key)]+[options[i][v][1] for i,v in zip(right,suffix)]
                checks.append(str(solver.check(*constraints)))
            assert checks==['sat','unsat'],checks
            witness={'left_A':[(cells[i],options[i][v][0]) for i,v in zip(left,a)],
                     'left_B':[(cells[i],options[i][v][0]) for i,v in zip(left,b)],
                     'right':[(cells[i],options[i][v][0]) for i,v in zip(right,suffix)],
                     'check_A_right':checks[0],'check_B_right':checks[1],
                     'both_prefixes_extendable':bool(sa and sb)}
        reports.append({'summary':name,'states':len(groups[name]),'live_states':len(livegroups[name]),
                        'safe_on_enumerated_universe':hit is None,
                        'safe_on_extendable_prefixes':hit is None or not (hit[2] and hit[3]),'counterexample':witness})
    return {'cut':cut,'prefix_cartesian_size':universe_size,'prefixes_after_local_budget_input_filters':prefixes,
            'extendable_prefixes':local_prefixes,'exact_residual_states':len(residuals),
            'live_exact_residual_states':len(live_residuals),
            'live_cost_residual_states':len(cost_residuals),
            'live_compression':local_prefixes/len(live_residuals) if live_residuals else None,
            'summaries':reports}


def decision_diagram(cells, options, layouts, order):
    """Reduced multi-valued decision diagram, false=0 and true=1.
    Entire legal choice domains, not just observed values, define each node.
    Skipped variables are allowed iff all of their children agree.
    """
    rows=frozenset(tuple(row[i] for i in order) for row in layouts)
    unique={}; cache={}; trie_nodes=0
    def build(depth, remaining):
        nonlocal trie_nodes
        if not remaining:return 0
        if depth==len(order):return 1
        trie_nodes+=1
        key=(depth,remaining)
        if key in cache:return cache[key]
        children=[]
        for value in range(len(options[order[depth]])):
            tail=frozenset(r[1:] for r in remaining if r[0]==value)
            children.append(build(depth+1,tail))
        if len(set(children))==1:node=children[0]
        else:
            shape=(depth,tuple(children))
            if shape not in unique:unique[shape]=len(unique)+2
            node=unique[shape]
        cache[key]=node
        return node
    root=build(0,rows)
    # Compare with the non-sharing accepted-prefix trie (reject leaves implicit).
    raw=sum(len({r[:d] for r in rows}) for d in range(len(order)))
    # Evaluate every physical combination when small, otherwise every accepted
    # layout plus one-step mutations. Exact count below covers the whole domain.
    inverse={node:shape for shape,node in unique.items()}
    def count(node,depth):
        if node==0:return 0
        if node==1:return math.prod(len(options[i]) for i in order[depth:])
        at,children=inverse[node]
        skip=math.prod(len(options[i]) for i in order[depth:at])
        return skip*sum(count(c,at+1) for c in children)
    total=count(root,0)
    assert total==len(layouts),(total,len(layouts))
    def accepts(row):
        node=root
        while node>1:
            depth,children=inverse[node]; node=children[row[order[depth]]]
        return node==1
    assert all(accepts(row) for row in layouts)
    # Equal count + every enumerated row accepted proves exact relation equality.
    return {'order':[cells[i] for i in order],'trie_nonterminal_nodes':raw,
            'mdd_nonterminal_nodes':len(unique),'compression':raw/len(unique),
            'accepted_assignment_count':total}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'target/boundary-lab')
    ap.add_argument('--seconds',type=float,default=60); ap.add_argument('--limit',type=int,default=10000)
    ap.add_argument('--only',default=''); args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    suite=[('inv3',(1,3,2),'inv',3),('inv4',(1,4,2),'inv',4),
           ('inv5',(1,4,2),'inv',5),('nor4',(1,4,2),'nor',4),('nor5',(1,4,2),'nor',5)]
    results=[]
    for name,dims,mode,budget in suite:
        if args.only and name not in args.only.split(','):continue
        prefix=out/name; ip=out/(name+'.rsdsl'); ip.write_text(instance(name,dims,mode,budget))
        proc=subprocess.run([str(ROOT/'target/release/examples/boundary_export'),str(MODEL),str(ip),str(prefix),*map(str,dims)],capture_output=True,text=True)
        if proc.returncode:raise RuntimeError(proc.stderr)
        solver,cells,options=load(prefix)
        print(name,'enumerating',flush=True)
        layouts,status,elapsed=enumerate_layouts(solver,options,args.limit,args.seconds)
        row={'instance':name,'dimensions':dims,'max_blocks':budget,'model_count':len(layouts),
             'enumeration_status':status,'enumeration_seconds':elapsed,'cuts':[],
             'instance_sha256':hashlib.sha256(ip.read_bytes()).hexdigest()}
        # Never certify a continuation set from a partial enumeration.
        if status=='exhausted' and layouts:
            orders={'spatial':list(range(len(cells))),
                    'reverse':list(reversed(range(len(cells)))),
                    'height_first':sorted(range(len(cells)),key=lambda i:(cells[i][2],cells[i][1],cells[i][0]))}
            row['decision_diagrams']={name:decision_diagram(cells,options,layouts,order) for name,order in orders.items()}
            for cut in range(1,dims[1]):
                result=analyze(solver,cells,options,layouts,cut,budget); row['cuts'].append(result)
                print(name,'cut',cut,{k:v for k,v in result.items() if k!='summaries'},flush=True)
                for s in result.get('summaries',[]): print(' ',s['summary'],s['live_states'],s['safe_on_enumerated_universe'],flush=True)
        results.append(row)
        (out/'results.json').write_text(json.dumps({'z3':z3.get_version_string(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'scope':'projected physical layouts of static rsdsl; all signal/order auxiliaries existential', 'results':results},ensure_ascii=False,indent=2))
        (out/(name+'.layouts.json')).write_text(json.dumps({'cells':cells,'options':[[v for v,_ in x] for x in options],'layouts':layouts}))
    print('Results:',out/'results.json',flush=True)
if __name__=='__main__':main()
