from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

@dataclass
class ComplexData:
    features: np.ndarray
    labels: np.ndarray
    patient_ids: list[str]
    edges: np.ndarray
    triangles: np.ndarray
    adjacency: sparse.csr_matrix
    coface: sparse.csr_matrix


def build_knn_edges(features: np.ndarray, k:int=12, metric:str='cosine') -> np.ndarray:
    n=features.shape[0]
    k=min(max(1,int(k)), n-1)
    nn=NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(features)
    idx=nn.kneighbors(features, return_distance=False)[:,1:]
    edges=set()
    for i,row in enumerate(idx):
        for j in row:
            a,b=sorted((i,int(j))); edges.add((a,b))
    return np.array(sorted(edges), dtype=np.int64)


def flag_triangles(n:int, edges:np.ndarray, max_triangles:int|None=None) -> np.ndarray:
    neigh=[set() for _ in range(n)]
    for a,b in edges:
        neigh[int(a)].add(int(b)); neigh[int(b)].add(int(a))
    tris=[]
    for i in range(n):
        ns=sorted([j for j in neigh[i] if j>i])
        for j,k in combinations(ns,2):
            if k in neigh[j]:
                tris.append((i,j,k))
                if max_triangles and len(tris)>=max_triangles:
                    return np.array(tris,dtype=np.int64)
    return np.array(tris,dtype=np.int64).reshape((-1,3)) if tris else np.empty((0,3),dtype=np.int64)


def norm_adj(n:int, pairs:np.ndarray) -> sparse.csr_matrix:
    if pairs.size==0:
        return sparse.eye(n,format='csr')
    row=np.r_[pairs[:,0], pairs[:,1], np.arange(n)]
    col=np.r_[pairs[:,1], pairs[:,0], np.arange(n)]
    data=np.ones(len(row), dtype=np.float32)
    A=sparse.coo_matrix((data,(row,col)), shape=(n,n)).tocsr()
    A.data[:] = 1.0
    d=np.asarray(A.sum(axis=1)).ravel()
    d_inv=np.power(d, -0.5, where=d>0); d_inv[d==0]=0
    D=sparse.diags(d_inv)
    return (D@A@D).tocsr()


def coface_from_triangles(n:int, triangles:np.ndarray) -> sparse.csr_matrix:
    if triangles.size==0:
        return sparse.eye(n,format='csr')
    pairs=[]
    for tri in triangles:
        for a,b in combinations(map(int,tri),2):
            pairs.append((a,b)); pairs.append((b,a))
    pairs.extend((i,i) for i in range(n))
    row,col=zip(*pairs)
    C=sparse.coo_matrix((np.ones(len(row),dtype=np.float32),(row,col)),shape=(n,n)).tocsr()
    d=np.asarray(C.sum(axis=1)).ravel()
    d_inv=np.power(d, -0.5, where=d>0); d_inv[d==0]=0
    D=sparse.diags(d_inv)
    return (D@C@D).tocsr()


def save_complex(features, labels, patient_ids, out_dir, k=12, metric='cosine'):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    edges=build_knn_edges(features,k=k,metric=metric)
    triangles=flag_triangles(features.shape[0],edges)
    A=norm_adj(features.shape[0],edges)
    C=coface_from_triangles(features.shape[0],triangles)
    np.save(out/'features.npy', features.astype('float32'))
    np.save(out/'labels.npy', labels.astype('int64'))
    np.save(out/'edges.npy', edges)
    np.save(out/'triangles.npy', triangles)
    sparse.save_npz(out/'adjacency_norm.npz', A)
    sparse.save_npz(out/'coface_adjacency_norm.npz', C)
    pd.DataFrame({'patient_id':patient_ids}).to_csv(out/'patient_ids.csv', index=False)
    pd.DataFrame([{
        'n_patients':features.shape[0], 'n_features':features.shape[1], 'k':k,
        'n_edges':len(edges), 'n_triangles':len(triangles),
        'edge_density': 2*len(edges)/(features.shape[0]*(features.shape[0]-1)),
        'mean_triangles_per_patient': 3*len(triangles)/features.shape[0] if features.shape[0] else 0
    }]).to_csv(out/'complex_summary.csv', index=False)


def load_complex(path)->ComplexData:
    p=Path(path)
    return ComplexData(
        features=np.load(p/'features.npy'), labels=np.load(p/'labels.npy'),
        patient_ids=pd.read_csv(p/'patient_ids.csv')['patient_id'].tolist(),
        edges=np.load(p/'edges.npy'), triangles=np.load(p/'triangles.npy'),
        adjacency=sparse.load_npz(p/'adjacency_norm.npz'),
        coface=sparse.load_npz(p/'coface_adjacency_norm.npz'))
