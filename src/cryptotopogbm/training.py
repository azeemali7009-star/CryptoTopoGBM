from __future__ import annotations
import copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .metrics import binary_metrics, concordance_index
from .models import sparse_to_torch

def make_splits(labels, test_size=0.2, validation_size=0.2, seed=7):
    idx=np.arange(len(labels))
    trainval,test=train_test_split(idx,test_size=test_size,random_state=seed,stratify=labels)
    rel_val=validation_size/(1-test_size)
    train,val=train_test_split(trainval,test_size=rel_val,random_state=seed,stratify=labels[trainval])
    return train,val,test

def train_one(model,x,y,adj,cof,train_idx,val_idx,epochs=220,lr=1e-3,weight_decay=1e-4,device='cpu'):
    model=model.to(device); x=x.to(device); y=y.to(device)
    adj=adj.to(device) if adj is not None else None; cof=cof.to(device) if cof is not None else None
    train_idx=torch.tensor(train_idx,dtype=torch.long,device=device); val_idx=torch.tensor(val_idx,dtype=torch.long,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay); loss_fn=torch.nn.BCEWithLogitsLoss()
    best_state=None; best_val=float('inf'); patience=50; stale=0
    for epoch in range(int(epochs)):
        model.train(); opt.zero_grad(); loss=loss_fn(model(x,adj,cof)[train_idx], y[train_idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val_loss=loss_fn(model(x,adj,cof)[val_idx], y[val_idx]).item()
        if val_loss < best_val-1e-5:
            best_val=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale += 1
            if stale>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def predict(model,x,adj,cof,idx,device='cpu'):
    model.eval(); model=model.to(device); x=x.to(device)
    adj=adj.to(device) if adj is not None else None; cof=cof.to(device) if cof is not None else None
    with torch.no_grad():
        logits=model(x,adj,cof).detach().cpu().numpy()
    return 1/(1+np.exp(-logits[idx])), logits[idx]

def evaluate(model,x,y,adj,cof,idx,times=None,events=None,device='cpu'):
    prob,logits=predict(model,x,adj,cof,idx,device=device)
    out=binary_metrics(y.detach().cpu().numpy()[idx],prob)
    if times is not None and events is not None:
        out['c_index']=concordance_index(np.asarray(times)[idx], logits, np.asarray(events)[idx])
    return out

def extract_embeddings(model,x,adj,cof,device='cpu'):
    model.eval(); model=model.to(device); x=x.to(device)
    adj=adj.to(device) if adj is not None else None; cof=cof.to(device) if cof is not None else None
    with torch.no_grad(): logits,z=model(x,adj,cof,return_embedding=True)
    return logits.detach().cpu().numpy(), z.detach().cpu().numpy()
