#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def make_demo(n=360, p_rna=260, p_meth=180, p_cnv=120, seed=7):
    rng=np.random.default_rng(seed)
    p=p_rna+p_meth+p_cnv
    # Latent molecular axes: proneural-classical-mesenchymal-like, hypoxia/immune, cell-cycle.
    centers=np.array([[-1.1,0.0,0.4],[0.2,-0.9,-0.3],[1.0,0.8,0.7]])
    subtype=rng.choice(3,size=n,p=[0.32,0.36,0.32])
    latent=centers[subtype]+rng.normal(scale=[0.55,0.50,0.45], size=(n,3))
    load=rng.normal(scale=0.28,size=(3,p))
    x=latent@load + rng.normal(scale=1.05,size=(n,p))
    # Inject weak pathway signals across omics blocks.
    x[:,0:25] += (subtype==0)[:,None]*1.00
    x[:,25:50] += (subtype==1)[:,None]*0.95
    x[:,50:75] += (subtype==2)[:,None]*1.05
    x[:,p_rna:p_rna+30] += latent[:,1:2]*0.75
    x[:,p_rna+p_meth:p_rna+p_meth+30] += (latent[:,2:3]>0).astype(float)*0.70
    # Construct a hidden local topology signal in latent space: dense high-risk molecular neighbourhoods.
    nn=NearestNeighbors(n_neighbors=13,metric='euclidean').fit(latent)
    dist,idx=nn.kneighbors(latent)
    local_mes=np.mean((subtype[idx[:,1:]]==2).astype(float),axis=1)
    local_density=1/(dist[:,1:].mean(axis=1)+1e-6)
    density_z=(local_density-local_density.mean())/local_density.std()
    raw=0.18*latent[:,0]+0.10*latent[:,1]+0.06*x[:,12]-0.04*x[:,35]+1.45*local_mes+0.95*density_z+rng.normal(scale=0.70,size=n)
    prob=1/(1+np.exp(-raw))
    label=(prob>=np.quantile(prob,0.50)).astype(int)
    event=rng.binomial(1,0.74,size=n)
    days=np.maximum(45, 900-390*prob + rng.normal(scale=120,size=n)).astype(int)
    cols=[f'RNA_{i:04d}' for i in range(p_rna)]+[f'METH_{i:04d}' for i in range(p_meth)]+[f'CNV_{i:04d}' for i in range(p_cnv)]
    df=pd.DataFrame(x,columns=cols)
    df.insert(0,'patient_id',[f'DEMO-GBM-{i:04d}' for i in range(n)])
    df['survival_days']=days; df['event']=event; df['poor_survival_label']=label; df['molecular_subtype']=subtype
    df['latent_topology_risk']=prob
    return df

def main():
    ap=argparse.ArgumentParser(description='Create a GBM-like multi-omics demonstration dataset for pipeline validation only.')
    ap.add_argument('--out',default='data/demo/gbm_like_multiomics_demo.csv'); ap.add_argument('--n',type=int,default=360); ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args(); Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    df=make_demo(n=args.n,seed=args.seed); df.to_csv(args.out,index=False)
    print(f'Wrote {args.out} with shape {df.shape}. This is synthetic demonstration data, not TCGA.')
if __name__=='__main__': main()
