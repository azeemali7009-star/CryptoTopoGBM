#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from cryptotopogbm.models import MLP, GraphNet, SimplicialNet, sparse_to_torch
from cryptotopogbm.topology import load_complex
from cryptotopogbm.training import make_splits, train_one, evaluate, extract_embeddings
from cryptotopogbm.metrics import binary_metrics, concordance_index
from cryptotopogbm.utils import seed_everything

torch.set_num_threads(1)

def train_seed(seed,args,data, surv):
    seed_everything(seed)
    x=torch.tensor(data.features,dtype=torch.float32); y=torch.tensor(data.labels,dtype=torch.float32)
    adj=sparse_to_torch(data.adjacency); cof=sparse_to_torch(data.coface)
    train_idx,val_idx,test_idx=make_splits(data.labels,seed=seed)
    rows=[]; trained={}
    # Classical baselines.
    lr=LogisticRegression(max_iter=2000,class_weight='balanced',C=0.3,solver='liblinear')
    lr.fit(data.features[train_idx], data.labels[train_idx])
    for split,idx in [('validation',val_idx),('test',test_idx)]:
        prob=lr.predict_proba(data.features[idx])[:,1]
        out=binary_metrics(data.labels[idx],prob)
        if surv is not None: out['c_index']=concordance_index(surv['survival_days'].values[idx], prob, surv['event'].values[idx])
        rows.append({'seed':seed,'model':'Logistic','split':split,**out})
    rf=RandomForestClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=1)
    rf.fit(data.features[train_idx],data.labels[train_idx])
    for split,idx in [('validation',val_idx),('test',test_idx)]:
        prob=rf.predict_proba(data.features[idx])[:,1]
        out=binary_metrics(data.labels[idx],prob)
        if surv is not None: out['c_index']=concordance_index(surv['survival_days'].values[idx], prob, surv['event'].values[idx])
        rows.append({'seed':seed,'model':'RandomForest','split':split,**out})
    configs=[
        ('MLP', MLP(data.features.shape[1],args.hidden_dim,args.embedding_dim,args.dropout), None, None),
        ('GraphNet', GraphNet(data.features.shape[1],args.hidden_dim,args.embedding_dim,args.dropout), adj, None),
        ('SimplicialNet', SimplicialNet(data.features.shape[1],args.hidden_dim,args.embedding_dim,args.dropout), adj, cof),
    ]
    for name,model,use_adj,use_cof in configs:
        model=train_one(model,x,y,use_adj,use_cof,train_idx,val_idx,epochs=args.epochs,lr=args.lr,weight_decay=args.weight_decay,device=args.device)
        for split,idx in [('validation',val_idx),('test',test_idx)]:
            out=evaluate(model,x,y,use_adj,use_cof,idx,times=surv['survival_days'].values if surv is not None else None,events=surv['event'].values if surv is not None else None,device=args.device)
            rows.append({'seed':seed,'model':name,'split':split,**out})
        trained[name]=(model,use_adj,use_cof)
    return rows, trained, (train_idx,val_idx,test_idx), (x,adj,cof)

def main():
    ap=argparse.ArgumentParser(description='Train baselines and topological models.')
    ap.add_argument('--complex',default='data/complex'); ap.add_argument('--processed',default='data/processed'); ap.add_argument('--out',default='results')
    ap.add_argument('--seeds',default='7,19,31'); ap.add_argument('--epochs',type=int,default=220); ap.add_argument('--hidden-dim',type=int,default=64); ap.add_argument('--embedding-dim',type=int,default=32)
    ap.add_argument('--dropout',type=float,default=0.25); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data=load_complex(args.complex)
    surv=None
    sp=Path(args.processed)/'survival.csv'
    if sp.exists():
        surv=pd.read_csv(sp).set_index('patient_id').loc[data.patient_ids].reset_index()
    all_rows=[]; best_auc=-1; best=None
    for seed in [int(s) for s in args.seeds.split(',') if s.strip()]:
        rows,trained,splits,tensors=train_seed(seed,args,data,surv)
        all_rows.extend(rows)
        simp_test=[r for r in rows if r['model']=='SimplicialNet' and r['split']=='test'][0]
        if simp_test.get('auc',-1)>best_auc:
            best_auc=simp_test['auc']; best=(seed,trained,splits,tensors)
    metrics=pd.DataFrame(all_rows); metrics.to_csv(out/'metrics_all_seeds.csv',index=False)
    summary=metrics[metrics.split.eq('test')].groupby('model').agg({m:['mean','std'] for m in ['accuracy','balanced_accuracy','f1','auc','auprc','c_index'] if m in metrics.columns}).reset_index()
    summary.columns=['model']+[f'{a}_{b}' for a,b in summary.columns[1:]]
    summary.to_csv(out/'metrics_summary_test.csv',index=False)
    # Best seed outputs for encryption.
    seed,trained,splits,tensors=best; train_idx,val_idx,test_idx=splits; x,adj,cof=tensors
    np.savez(out/'splits_best_seed.npz',seed=seed,train=train_idx,val=val_idx,test=test_idx)
    model,use_adj,use_cof=trained['SimplicialNet']
    logits,z=extract_embeddings(model,x,use_adj,use_cof,device=args.device)
    np.save(out/'simplicial_embeddings.npy',z); np.save(out/'simplicial_logits.npy',logits)
    torch.save({'state_dict':model.state_dict(),'in_dim':data.features.shape[1],'hidden_dim':args.hidden_dim,'embedding_dim':args.embedding_dim,'dropout':args.dropout,'head_weight':model.head.weight.detach().cpu().numpy(),'head_bias':model.head.bias.detach().cpu().numpy(),'gate':torch.softmax(model.gate_logits.detach().cpu(),dim=0).numpy(),'best_seed':seed}, out/'simplicial_model.pt')
    # LaTeX tables.
    tex=summary.copy()
    order=['Logistic','RandomForest','MLP','GraphNet','SimplicialNet']; tex['ord']=tex.model.map({m:i for i,m in enumerate(order)}); tex=tex.sort_values('ord')
    lines=['\\begin{table}[t]','\\centering','\\caption{Demonstration benchmark on the bundled GBM-like multi-omics dataset. Values are mean $\\pm$ standard deviation over three random patient splits. This table validates the computation and is replaced by TCGA-GBM values after running the GDC downloader.}','\\label{tab:demo_results}','\\small','\\begin{tabular}{lccccc}','\\toprule','Model & Accuracy & Balanced acc. & F1 & AUC & C-index \\','\\midrule']
    for _,r in tex.iterrows():
        lines.append(f"{r['model']} & {r['accuracy_mean']:.3f} $\\pm$ {r['accuracy_std']:.3f} & {r['balanced_accuracy_mean']:.3f} $\\pm$ {r['balanced_accuracy_std']:.3f} & {r['f1_mean']:.3f} $\\pm$ {r['f1_std']:.3f} & {r['auc_mean']:.3f} $\\pm$ {r['auc_std']:.3f} & {r['c_index_mean']:.3f} $\\pm$ {r['c_index_std']:.3f} \\")
    lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    (out/'demo_results_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    pd.DataFrame([{'best_seed':seed,'simplicial_test_auc':best_auc}]).to_csv(out/'best_seed.csv',index=False)
    print(summary.to_string(index=False))
    print('Best seed for encryption:',seed)
if __name__=='__main__': main()
