#!/usr/bin/env python
from __future__ import annotations
import argparse, math, secrets, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def sigmoid(x): return 1/(1+np.exp(-x))

def egcd(a,b):
    if b==0: return (a,1,0)
    g,x1,y1=egcd(b,a%b); return (g,y1,x1-(a//b)*y1)

def invmod(a,m):
    g,x,y=egcd(a%m,m)
    if g!=1: raise ValueError('not invertible')
    return x%m

def is_probable_prime(n,k=16):
    if n<2: return False
    small=[2,3,5,7,11,13,17,19,23,29,31,37]
    if n in small: return True
    if any(n%p==0 for p in small): return False
    d=n-1; s=0
    while d%2==0: s+=1; d//=2
    for _ in range(k):
        a=secrets.randbelow(n-3)+2
        x=pow(a,d,n)
        if x in (1,n-1): continue
        for _ in range(s-1):
            x=pow(x,2,n)
            if x==n-1: break
        else: return False
    return True

def rand_prime(bits):
    while True:
        p=secrets.randbits(bits) | (1<<(bits-1)) | 1
        if is_probable_prime(p): return p

def lcm(a,b): return abs(a*b)//math.gcd(a,b)

class Paillier:
    def __init__(self,bits=512):
        p=rand_prime(bits//2); q=rand_prime(bits//2)
        while q==p: q=rand_prime(bits//2)
        self.n=p*q; self.n2=self.n*self.n; self.g=self.n+1
        self.lam=lcm(p-1,q-1)
        u=pow(self.g,self.lam,self.n2); L=(u-1)//self.n; self.mu=invmod(L,self.n)
    def encode(self,m:int): return m % self.n
    def decode(self,m:int): return m-self.n if m>self.n//2 else m
    def encrypt_int(self,m:int):
        m=self.encode(int(m)); r=secrets.randbelow(self.n-1)+1
        while math.gcd(r,self.n)!=1: r=secrets.randbelow(self.n-1)+1
        return (pow(self.g,m,self.n2)*pow(r,self.n,self.n2))%self.n2
    def decrypt_int(self,c:int):
        u=pow(c,self.lam,self.n2); L=(u-1)//self.n; return self.decode((L*self.mu)%self.n)
    def eadd(self,c1,c2): return (c1*c2)%self.n2
    def emul_plain(self,c,a:int):
        a=int(a)
        if a>=0: return pow(c,a,self.n2)
        return invmod(pow(c,-a,self.n2), self.n2)

def encrypted_dot(pk:Paillier, enc_vec, w_int, bias_int):
    acc=pk.encrypt_int(bias_int)
    for c,a in zip(enc_vec,w_int):
        if a: acc=pk.eadd(acc, pk.emul_plain(c,int(a)))
    return acc

def main():
    ap=argparse.ArgumentParser(description='Dependency-free Paillier encrypted inference for the final linear risk head.')
    ap.add_argument('--model',default='results/simplicial_model.pt'); ap.add_argument('--embeddings',default='results/simplicial_embeddings.npy')
    ap.add_argument('--out',default='results/paillier_encrypted_inference.csv'); ap.add_argument('--n-examples',type=int,default=20); ap.add_argument('--bits',type=int,default=512); ap.add_argument('--scale',type=int,default=10000)
    args=ap.parse_args(); Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.model,map_location='cpu',weights_only=False)
    z=np.load(args.embeddings).astype(float); w=ckpt['head_weight'].reshape(-1).astype(float); b=float(ckpt['head_bias'].reshape(-1)[0])
    n=min(args.n_examples,z.shape[0]); scale=args.scale
    t0=time.perf_counter(); pk=Paillier(bits=args.bits); keygen=time.perf_counter()-t0
    w_int=np.round(w*scale).astype(int); bias_int=int(round(b*scale*scale))
    rows=[]
    for i in range(n):
        plain=float(z[i].dot(w)+b)
        z_int=np.round(z[i]*scale).astype(int)
        t1=time.perf_counter(); enc=[pk.encrypt_int(int(v)) for v in z_int]; enc_time=time.perf_counter()-t1
        t2=time.perf_counter(); c=encrypted_dot(pk,enc,w_int,bias_int); eval_time=time.perf_counter()-t2
        t3=time.perf_counter(); dec_int=pk.decrypt_int(c); dec_time=time.perf_counter()-t3
        dec=dec_int/(scale*scale)
        rows.append({'example_index':i,'plain_logit':plain,'paillier_decrypted_logit':dec,'absolute_error':abs(plain-dec),'plain_probability':float(sigmoid(plain)),'paillier_probability':float(sigmoid(dec)),'keygen_seconds':keygen if i==0 else 0.0,'encrypt_embedding_seconds':enc_time,'encrypted_dot_seconds':eval_time,'decrypt_seconds':dec_time,'key_bits':args.bits,'scale':scale})
    df=pd.DataFrame(rows); df.to_csv(args.out,index=False)
    summary=pd.DataFrame([{'n_examples':n,'key_bits':args.bits,'mean_absolute_logit_error':df.absolute_error.mean(),'max_absolute_logit_error':df.absolute_error.max(),'mean_encrypt_seconds':df.encrypt_embedding_seconds.mean(),'mean_encrypted_dot_seconds':df.encrypted_dot_seconds.mean(),'mean_decrypt_seconds':df.decrypt_seconds.mean()}])
    summary.to_csv(Path(args.out).with_name('paillier_encrypted_summary.csv'),index=False)
    lines=['\\begin{table}[t]','\\centering','\\caption{Encrypted final-head inference with dependency-free Paillier additive homomorphic encryption on the bundled benchmark. The encrypted computation evaluates the same linear risk head on quantized encrypted embeddings.}','\\label{tab:crypto_results}','\\small','\\begin{tabular}{lcc}','\\toprule','Quantity & Value & Unit \\','\\midrule']
    r=summary.iloc[0]
    lines.append(f"Encrypted test examples & {int(r.n_examples)} & patients \\")
    lines.append(f"Key size & {int(r.key_bits)} & bits \\")
    lines.append(f"Mean absolute logit error & {r.mean_absolute_logit_error:.6f} & logit \\")
    lines.append(f"Maximum absolute logit error & {r.max_absolute_logit_error:.6f} & logit \\")
    lines.append(f"Mean embedding encryption time & {r.mean_encrypt_seconds:.4f} & s/patient \\")
    lines.append(f"Mean encrypted dot-product time & {r.mean_encrypted_dot_seconds:.4f} & s/patient \\")
    lines.append(f"Mean decryption time & {r.mean_decrypt_seconds:.6f} & s/patient \\")
    lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    (Path(args.out).with_name('crypto_results_table.tex')).write_text('\n'.join(lines),encoding='utf-8')
    print(summary.to_string(index=False))
if __name__=='__main__': main()
