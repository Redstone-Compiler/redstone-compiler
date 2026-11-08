#!/usr/bin/env python3
"""Generate fixed-pin holdout cases and independent direct-AllSAT references."""
import json
from pathlib import Path
import re
import subprocess
from run import ROOT, MODEL, instance
from online_boundary import read_problem, search


def main():
    out=ROOT/'target/boundary-lab';out.mkdir(parents=True,exist_ok=True)
    fixtures=ROOT/'tools/pruning_lab/boundary_fixtures'
    suite=[('inv_fixed6',(1,4,3),'inv',6),
           ('inv_fixed7',(1,5,3),'inv',7),
           ('nor_fixed7',(2,4,2),'nor',7),
           ('inv_fixed8',(1,4,3),'inv',8),
           ('inv_long9',(1,5,3),'inv',9),
           ('nor_fixed10',(2,4,3),'nor',10),
           ('inv_long10',(1,5,3),'inv',10),
           ('inv_wide8',(2,4,3),'inv',8)]
    records=[]
    for name,dims,mode,bound in suite:
        text=instance(name,dims,mode,bound)
        sites='(Cell(0,0,1),Floor,a)'
        if mode=='nor':sites+=',(Cell(1,0,1),Floor,b)'
        text=re.sub(r'fact switch_site = .*?;',f'fact switch_site = [{sites}];',text)
        text=re.sub(r'fact output_site = .*?;',
                    f'fact output_site = [(out,Cell(0,{dims[1]-1},1))];',text)
        prefix=out/name;ip=prefix.with_suffix('.rsdsl');ip.write_text(text)
        proc=subprocess.run([str(ROOT/'target/release/examples/boundary_export'),str(MODEL),str(ip),str(prefix),*map(str,dims)],capture_output=True,text=True)
        if proc.returncode:raise RuntimeError(proc.stderr)
        cells,options,_=read_problem(prefix)
        row,layouts=search(prefix,'allsat',19,45,0,'native');row['instance']=name
        row.update(dimensions=dims,max_blocks=bound)
        records.append(row);print(json.dumps(row),flush=True)
        if layouts is not None:
            (fixtures/(name+'.layouts.json')).write_text(json.dumps({
                'cells':cells,'options':[[label for label,_ in opts] for opts in options],
                'layouts':layouts}))
        (out/'extended_reference_results.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
