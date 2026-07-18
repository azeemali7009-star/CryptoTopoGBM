#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from cryptotopogbm.topology import load_complex

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'manuscript'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
RES=ROOT/'results'

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

def workflow():
    fig,ax=plt.subplots(figsize=(11,3.0)); ax.axis('off')
    boxes=[('Open data\nTCGA-GBM / TCGA-LGG\nRNA-seq + clinical survival',0.02),('Patient similarity\n$k$NN graph + flag\n2-simplicial complex',0.26),('Topo encoder\nself + edge + triangle\nmessage passing',0.50),('Privacy layer\nHE-encrypted final\nrisk head',0.74)]
    for text,x in boxes:
        rect=patches.FancyBboxPatch((x,0.25),0.21,0.50,boxstyle='round,pad=0.02',linewidth=1.2,edgecolor='black',facecolor='white')
        ax.add_patch(rect); ax.text(x+0.105,0.50,text,ha='center',va='center',fontsize=10)
    for x in [0.23,0.47,0.71]:
        ax.annotate('',xy=(x+0.025,0.50),xytext=(x,0.50),arrowprops=dict(arrowstyle='->',lw=1.5))
    ax.text(0.5,0.08,'Novelty: higher-order patient topology is learned locally; only encrypted topological embeddings are sent for outsourced risk scoring.',ha='center',fontsize=9)
    save(fig,'figure1_workflow')

def topology():
    data=load_complex(ROOT/'data'/'complex')
    coords=PCA(n_components=2,random_state=7).fit_transform(data.features)
    lab=data.labels
    fig,ax=plt.subplots(figsize=(6.8,5.2)); ax.axis('off')
    # draw subset for readability
    subset=np.arange(min(75,len(coords)))
    edge_set=[e for e in data.edges if e[0] in subset and e[1] in subset]
    tri_set=[t for t in data.triangles if all(v in subset for v in t)][:65]
    xy=coords[subset]; xy=(xy-xy.min(0))/(xy.max(0)-xy.min(0)+1e-9)
    for tri in tri_set:
        pts=xy[list(tri)]
        ax.add_patch(patches.Polygon(pts,closed=True,alpha=0.10,edgecolor=None))
    for a,b in edge_set:
        ax.plot([xy[a,0],xy[b,0]],[xy[a,1],xy[b,1]],lw=0.5,alpha=0.35)
    ax.scatter(xy[:,0],xy[:,1],c=lab[subset],s=38,edgecolors='black',linewidths=0.4)
    ax.set_title('Patient similarity graph lifted to a 2-simplicial complex',fontsize=12)
    ax.text(0.02,-0.08,f"Bundled benchmark complex: {len(data.patient_ids)} patients, {len(data.edges)} edges, {len(data.triangles)} triangles",transform=ax.transAxes,fontsize=9)
    save(fig,'figure2_simplicial_complex')

def model():
    fig,ax=plt.subplots(figsize=(10,3.6)); ax.axis('off')
    labels=[('$X$',0.05,0.55),('$AX$',0.05,0.30),('$CX$',0.05,0.05),('Gated\nTopo block',0.30,0.30),('Embedding\n$z_i$',0.55,0.30),('HE risk head\n$Enc(z_i^T w+b)$',0.78,0.30)]
    for txt,x,y in labels:
        rect=patches.FancyBboxPatch((x,y),0.16,0.18,boxstyle='round,pad=0.02',edgecolor='black',facecolor='white')
        ax.add_patch(rect); ax.text(x+0.08,y+0.09,txt,ha='center',va='center',fontsize=10)
    for y in [0.64,0.39,0.14]: ax.annotate('',xy=(0.30,0.39),xytext=(0.21,y),arrowprops=dict(arrowstyle='->',lw=1.2))
    ax.annotate('',xy=(0.55,0.39),xytext=(0.46,0.39),arrowprops=dict(arrowstyle='->',lw=1.2))
    ax.annotate('',xy=(0.78,0.39),xytext=(0.71,0.39),arrowprops=dict(arrowstyle='->',lw=1.2))
    save(fig,'figure3_model')

def metrics():
    df=pd.read_csv(RES/'metrics_summary_test.csv')
    order=['Logistic','RandomForest','MLP','GraphNet','SimplicialNet']; df=df.set_index('model').loc[order].reset_index()
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.bar(df['model'],df['auc_mean'],yerr=df['auc_std'],capsize=4)
    ax.set_ylabel('Test AUC')
    ax.set_ylim(0.50,0.92)
    ax.set_title('Demonstration output: topology-aware models improve ranking performance')
    ax.tick_params(axis='x',rotation=25)
    for i,v in enumerate(df['auc_mean']): ax.text(i,v+0.01,f'{v:.3f}',ha='center',fontsize=8)
    save(fig,'figure4_results')

def crypto():
    df=pd.read_csv(RES/'paillier_encrypted_inference.csv')
    fig,ax=plt.subplots(figsize=(6.5,4))
    ax.scatter(df['plain_logit'],df['paillier_decrypted_logit'],s=30)
    mn=min(df['plain_logit'].min(),df['paillier_decrypted_logit'].min()); mx=max(df['plain_logit'].max(),df['paillier_decrypted_logit'].max())
    ax.plot([mn,mx],[mn,mx],lw=1)
    ax.set_xlabel('Plaintext logit')
    ax.set_ylabel('Decrypted Paillier logit')
    ax.set_title('Encrypted final-head inference preserves the risk score')
    ax.text(0.02,0.96,f"Mean absolute error = {df.absolute_error.mean():.6f}",transform=ax.transAxes,va='top',fontsize=9)
    save(fig,'figure5_crypto_results')

if __name__=='__main__':
    workflow(); topology(); model(); metrics(); crypto()
