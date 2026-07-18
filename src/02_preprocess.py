#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_demo(path:Path,out:Path,top_features:int):
    df=pd.read_csv(path)
    meta=['patient_id','survival_days','event','poor_survival_label','molecular_subtype','latent_topology_risk']
    features=df.drop(columns=[c for c in meta if c in df.columns])
    vars_=features.var(axis=0).sort_values(ascending=False)
    sel=vars_.head(top_features).index.tolist()
    x=StandardScaler().fit_transform(features[sel].values.astype(float))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(x,index=df['patient_id'],columns=sel).to_csv(out/'features.csv')
    pd.DataFrame({'patient_id':df['patient_id'],'label':df['poor_survival_label'].astype(int)}).to_csv(out/'labels.csv',index=False)
    pd.DataFrame({'patient_id':df['patient_id'],'survival_days':df['survival_days'],'event':df['event'],'molecular_subtype':df['molecular_subtype']}).to_csv(out/'survival.csv',index=False)
    pd.DataFrame({'feature':sel,'variance':vars_.loc[sel].values}).to_csv(out/'selected_features.csv',index=False)
    pd.DataFrame([{'source':'synthetic_gbm_like_demo','n_patients':len(df),'n_selected_features':len(sel),'n_events':int(df['event'].sum()),'positive_label_rate':float(df['poor_survival_label'].mean())}]).to_csv(out/'cohort_summary.csv',index=False)


def parse_star_file(path:Path):
    # GDC STAR-count files include gene_id, gene_name, gene_type and count/TPM columns.
    df=pd.read_csv(path,sep='\t',comment='#')
    gene_col='gene_name' if 'gene_name' in df.columns else df.columns[1]
    val_col=None
    for c in ['tpm_unstranded','fpkm_unstranded','unstranded']:
        if c in df.columns: val_col=c; break
    if val_col is None: val_col=df.columns[-1]
    s=df[[gene_col,val_col]].dropna().drop_duplicates(subset=[gene_col]).set_index(gene_col)[val_col].astype(float)
    return s


def preprocess_gdc(raw:Path,out:Path,top_features:int):
    manifest=pd.read_csv(raw/'manifest_flat.csv')
    expr=[]; ids=[]
    for _,r in manifest.iterrows():
        p=raw/'star_counts'/r['file_name']
        if not p.exists(): continue
        s=parse_star_file(p); expr.append(s); ids.append(r['case_submitter_id'])
    mat=pd.concat(expr,axis=1).T
    mat.index=ids
    mat=mat.groupby(level=0).mean()
    mat=np.log2(mat+1.0)
    # Clinical labels
    cases=json.loads((raw/'clinical_cases.json').read_text())
    clin=[]
    for c in cases:
        diag=(c.get('diagnoses') or [{}])[0]
        dtd=diag.get('days_to_death'); dlf=diag.get('days_to_last_follow_up')
        vital=str(diag.get('vital_status','')).lower()
        event=1 if vital=='dead' or dtd is not None else 0
        days=dtd if dtd is not None else dlf
        try: days=float(days)
        except Exception: days=np.nan
        clin.append({'patient_id':c.get('submitter_id'), 'survival_days':days, 'event':event})
    clin=pd.DataFrame(clin).dropna(subset=['survival_days'])
    both=mat.index.intersection(clin['patient_id'])
    mat=mat.loc[both]
    clin=clin.set_index('patient_id').loc[both].reset_index()
    med=np.median(clin.loc[clin.event.eq(1),'survival_days'])
    keep=~((clin.event.eq(0)) & (clin.survival_days<med))
    mat=mat.loc[clin.loc[keep,'patient_id']]
    clin=clin.loc[keep].copy()
    clin['label']=((clin.event.eq(1)) & (clin.survival_days<=med)).astype(int)
    vars_=mat.var(axis=0).sort_values(ascending=False); sel=vars_.head(top_features).index.tolist()
    x=StandardScaler().fit_transform(mat[sel].values.astype(float))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(x,index=mat.index,columns=sel).to_csv(out/'features.csv')
    clin[['patient_id','label']].to_csv(out/'labels.csv',index=False)
    clin[['patient_id','survival_days','event']].to_csv(out/'survival.csv',index=False)
    pd.DataFrame({'feature':sel,'variance':vars_.loc[sel].values}).to_csv(out/'selected_features.csv',index=False)
    pd.DataFrame([{'source':'TCGA_GBM_GDC','n_patients':len(clin),'n_selected_features':len(sel),'n_events':int(clin.event.sum()),'event_time_median':float(med),'positive_label_rate':float(clin.label.mean())}]).to_csv(out/'cohort_summary.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description='Preprocess demo or GDC TCGA-GBM expression and survival labels.')
    ap.add_argument('--mode',choices=['demo','gdc'],default='demo')
    ap.add_argument('--input',default='data/demo/gbm_like_multiomics_demo.csv')
    ap.add_argument('--out',default='data/processed'); ap.add_argument('--top-features',type=int,default=500)
    args=ap.parse_args()
    if args.mode=='demo': preprocess_demo(Path(args.input),Path(args.out),args.top_features)
    else: preprocess_gdc(Path(args.input),Path(args.out),args.top_features)
    print(f'Wrote processed files to {args.out}')
if __name__=='__main__': main()
