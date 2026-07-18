#!/usr/bin/env python
from __future__ import annotations
import argparse, gzip, io, json, time
from pathlib import Path
from typing import Any
import pandas as pd
import requests
from tqdm import tqdm
GDC='https://api.gdc.cancer.gov'

def post(endpoint:str,payload:dict[str,Any]):
    r=requests.post(f'{GDC}/{endpoint.lstrip("/")}',json=payload,timeout=90); r.raise_for_status(); return r.json()

def get(endpoint:str,params=None):
    r=requests.get(f'{GDC}/{endpoint.lstrip("/")}',params=params,timeout=180); r.raise_for_status(); return r

def filters(project_id='TCGA-GBM'):
    return {'op':'and','content':[
        {'op':'in','content':{'field':'cases.project.project_id','value':[project_id]}},
        {'op':'in','content':{'field':'files.data_category','value':['Transcriptome Profiling']}},
        {'op':'in','content':{'field':'files.data_type','value':['Gene Expression Quantification']}},
        {'op':'in','content':{'field':'files.analysis.workflow_type','value':['STAR - Counts']}},
        {'op':'in','content':{'field':'files.access','value':['open']}}
    ]}

def main():
    ap=argparse.ArgumentParser(description='Download open-access TCGA-GBM STAR-count expression files and clinical data from GDC.')
    ap.add_argument('--project',default='TCGA-GBM'); ap.add_argument('--out',default='data/raw/tcga_gbm'); ap.add_argument('--max-files',type=int,default=0)
    args=ap.parse_args(); out=Path(args.out); (out/'star_counts').mkdir(parents=True,exist_ok=True)
    payload={'filters':filters(args.project),'fields':'file_id,file_name,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type','format':'JSON','size':'2000'}
    hits=post('files',payload)['data']['hits']
    if args.max_files: hits=hits[:args.max_files]
    pd.DataFrame(hits).to_json(out/'gdc_file_manifest.json',orient='records',indent=2)
    rows=[]
    for h in tqdm(hits,desc='Downloading STAR-count files'):
        fid=h['file_id']; fname=h.get('file_name',fid+'.tsv')
        resp=get(f'data/{fid}')
        raw=resp.content
        if fname.endswith('.gz'):
            raw=gzip.decompress(raw)
        path=out/'star_counts'/fname.replace('/','_')
        path.write_bytes(raw)
        cases=h.get('cases',[]); case=cases[0] if cases else {}
        rows.append({'file_id':fid,'file_name':path.name,'case_submitter_id':case.get('submitter_id','')})
        time.sleep(0.05)
    pd.DataFrame(rows).to_csv(out/'manifest_flat.csv',index=False)
    clin=post('cases',{'filters':{'op':'in','content':{'field':'project.project_id','value':[args.project]}},'fields':'submitter_id,diagnoses.days_to_death,diagnoses.days_to_last_follow_up,diagnoses.vital_status,demographic.gender,demographic.age_at_index','format':'JSON','size':'2000'})
    Path(out/'clinical_cases.json').write_text(json.dumps(clin['data']['hits'],indent=2),encoding='utf-8')
    print(f'Downloaded {len(rows)} expression files to {out}.')
if __name__=='__main__': main()
