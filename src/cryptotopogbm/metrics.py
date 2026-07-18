from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score

def binary_metrics(y_true, y_prob):
    y_true=np.asarray(y_true).astype(int); y_prob=np.asarray(y_prob).astype(float)
    y_pred=(y_prob>=0.5).astype(int)
    out={
        'accuracy': float(accuracy_score(y_true,y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true,y_pred)),
        'f1': float(f1_score(y_true,y_pred,zero_division=0)),
    }
    try: out['auc']=float(roc_auc_score(y_true,y_prob))
    except Exception: out['auc']=float('nan')
    try: out['auprc']=float(average_precision_score(y_true,y_prob))
    except Exception: out['auprc']=float('nan')
    return out

def concordance_index(times, risks, events):
    times=np.asarray(times,float); risks=np.asarray(risks,float); events=np.asarray(events,int)
    permissible=0; concordant=0.0
    for i in range(len(times)):
        for j in range(i+1,len(times)):
            if times[i]==times[j]: continue
            if events[i]==1 and times[i]<times[j]:
                permissible+=1
                concordant += 1.0 if risks[i]>risks[j] else (0.5 if risks[i]==risks[j] else 0.0)
            elif events[j]==1 and times[j]<times[i]:
                permissible+=1
                concordant += 1.0 if risks[j]>risks[i] else (0.5 if risks[i]==risks[j] else 0.0)
    return float(concordant/permissible) if permissible else float('nan')
