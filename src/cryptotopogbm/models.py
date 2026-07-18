from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def sparse_to_torch(mat, dense_threshold:int=5000):
    if mat is None: return None
    if mat.shape[0] <= dense_threshold:
        return torch.tensor(mat.toarray(), dtype=torch.float32)
    mat=mat.tocoo().astype('float32')
    idx=torch.tensor(__import__('numpy').vstack([mat.row,mat.col]), dtype=torch.long)
    val=torch.tensor(mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx,val,torch.Size(mat.shape)).coalesce()

def opmm(op,x):
    if op is None: return x
    return torch.sparse.mm(op,x) if getattr(op,'is_sparse',False) else op@x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, embedding_dim=32, dropout=0.25):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,embedding_dim),nn.ReLU(),nn.Dropout(dropout))
        self.head=nn.Linear(embedding_dim,1)
    def forward(self,x,adj=None,cof=None,return_embedding=False):
        z=self.encoder(x); out=self.head(z).squeeze(-1)
        return (out,z) if return_embedding else out

class GraphNet(nn.Module):
    def __init__(self,in_dim,hidden_dim=64,embedding_dim=32,dropout=0.25):
        super().__init__()
        self.self0=nn.Linear(in_dim,hidden_dim,bias=False); self.edge0=nn.Linear(in_dim,hidden_dim,bias=True)
        self.self1=nn.Linear(hidden_dim,embedding_dim,bias=False); self.edge1=nn.Linear(hidden_dim,embedding_dim,bias=True)
        self.head=nn.Linear(embedding_dim,1); self.dropout=dropout
    def forward(self,x,adj,cof=None,return_embedding=False):
        h=F.relu(self.self0(x)+self.edge0(opmm(adj,x))); h=F.dropout(h,p=self.dropout,training=self.training)
        z=F.relu(self.self1(h)+self.edge1(opmm(adj,h))); z=F.dropout(z,p=self.dropout,training=self.training)
        out=self.head(z).squeeze(-1); return (out,z) if return_embedding else out

class SimplicialNet(nn.Module):
    def __init__(self,in_dim,hidden_dim=64,embedding_dim=32,dropout=0.25):
        super().__init__()
        self.self0=nn.Linear(in_dim,hidden_dim,bias=False); self.edge0=nn.Linear(in_dim,hidden_dim,bias=False); self.tri0=nn.Linear(in_dim,hidden_dim,bias=True)
        self.self1=nn.Linear(hidden_dim,embedding_dim,bias=False); self.edge1=nn.Linear(hidden_dim,embedding_dim,bias=False); self.tri1=nn.Linear(hidden_dim,embedding_dim,bias=True)
        self.head=nn.Linear(embedding_dim,1); self.dropout=dropout
        self.gate_logits=nn.Parameter(torch.tensor([0.0,0.0,0.25]))
    def forward(self,x,adj,cof,return_embedding=False):
        g=torch.softmax(self.gate_logits, dim=0)
        h=g[0]*self.self0(x)+g[1]*self.edge0(opmm(adj,x))+g[2]*self.tri0(opmm(cof,x))
        h=F.relu(h); h=F.dropout(h,p=self.dropout,training=self.training)
        z=g[0]*self.self1(h)+g[1]*self.edge1(opmm(adj,h))+g[2]*self.tri1(opmm(cof,h))
        z=F.relu(z); z=F.dropout(z,p=self.dropout,training=self.training)
        out=self.head(z).squeeze(-1); return (out,z) if return_embedding else out
