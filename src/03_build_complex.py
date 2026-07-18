#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from cryptotopogbm.topology import save_complex

def main():
    ap=argparse.ArgumentParser(description='Build patient kNN graph and 2-dimensional flag simplicial complex.')
    ap.add_argument('--processed',default='data/processed'); ap.add_argument('--out',default='data/complex')
    ap.add_argument('--k',type=int,default=12); ap.add_argument('--metric',default='cosine')
    args=ap.parse_args(); p=Path(args.processed)
    X=pd.read_csv(p/'features.csv',index_col=0); y=pd.read_csv(p/'labels.csv')
    ids=X.index.tolist(); lab=y.set_index('patient_id').loc[ids]['label'].values
    save_complex(X.values,lab,ids,args.out,k=args.k,metric=args.metric)
    print(pd.read_csv(Path(args.out)/'complex_summary.csv').to_string(index=False))
if __name__=='__main__': main()
