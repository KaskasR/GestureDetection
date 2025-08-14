# train_and_export_4class.py
# Train 4-class MLP (90->30->60->4) from windows.csv + feature_mapping.json,
# export nn_export/neuralnet_params.h (drop-in).

import os, json, csv, math
import numpy as np

WINDOW=15; H1=30; H2=60; OUT=4; SEED=42
WIN_CSV="windows.csv"; MAP_JSON="feature_mapping.json"
OUT_DIR="nn_export"; OUT_HDR=os.path.join(OUT_DIR,"neuralnet_params.h")

def relu(x): return np.maximum(x,0.)
def softmax(z):
    z=z-np.max(z,axis=1,keepdims=True); e=np.exp(z); return e/np.sum(e,axis=1,keepdims=True)
def one_hot(y,K):
    Y=np.zeros((len(y),K),np.float32); Y[np.arange(len(y)),y]=1.; return Y
def he(shape,rng): return rng.normal(0, math.sqrt(2/shape[1]), size=shape).astype(np.float32)

def xent(y_true,y_pred,eps=1e-8,weights=None):
    y_pred=np.clip(y_pred,eps,1-eps)
    ce=-np.sum(y_true*np.log(y_pred),axis=1)
    if weights is not None: ce*=weights
    return float(np.mean(ce))

def load_mapping():
    m=json.load(open(MAP_JSON,"r"))
    idx=np.array(m["selected_feature_indices"],int)
    mu =np.array(m["feature_mean"],np.float32)
    sd =np.array(m["feature_std"],np.float32)
    assert len(idx)==6 and len(mu)==6 and len(sd)==6
    return idx,mu,sd

def load_windows(sel_idx,mu6,sd6):
    X=[]; y=[]
    with open(WIN_CSV,"r",newline="") as f:
        for row in csv.reader(f):
            if not row: continue
            lab=int(float(row[0]))
            if lab not in (0,1,2,3): continue
            nfeat=int(float(row[1])); win=int(float(row[2]))
            if win!=WINDOW or nfeat<6: continue
            arr=np.array(list(map(float,row[3:])),np.float32)
            if arr.size!=win*nfeat: continue
            M=arr.reshape(WINDOW,nfeat)[:,sel_idx]                # (15,6)
            M = (M - mu6.reshape(1, 6)) / (sd6.reshape(1, 6) + 1e-8)  # per-feature norm         # per-feature norm
            X.append(M.reshape(-1))                                # (90,)
            y.append(lab)
    if not X: raise SystemExit("No samples for labels 0..3. Record each class first.")
    return np.stack(X,0), np.array(y,int)

def split_strat(X,y,val_ratio=0.2,seed=SEED):
    rng=np.random.default_rng(seed)
    Xtr=[]; ytr=[]; Xv=[]; yv=[]
    for c in sorted(set(y.tolist())):
        idx=np.where(y==c)[0]; rng.shuffle(idx)
        nv=max(1,int(len(idx)*val_ratio))
        vi,ti=idx[:nv],idx[nv:]
        Xv.append(X[vi]); yv.append(y[vi]); Xtr.append(X[ti]); ytr.append(y[ti])
    return np.vstack(Xtr),np.concatenate(ytr),np.vstack(Xv),np.concatenate(yv)

def class_weights(y):
    K=OUT; cnt=np.array([max(1,np.sum(y==k)) for k in range(K)],np.float32)
    return (cnt.sum()/(K*cnt))[y]

class MLP:
    def __init__(self,rng,in_dim):
        self.W0=he((H1,in_dim),rng); self.b0=np.zeros((1,H1),np.float32)
        self.W1=he((H2,H1),rng);     self.b1=np.zeros((1,H2),np.float32)
        self.W2=he((OUT,H2),rng);    self.b2=np.zeros((1,OUT),np.float32)
        self.m=[np.zeros_like(self.W0),np.zeros_like(self.b0),np.zeros_like(self.W1),
                np.zeros_like(self.b1),np.zeros_like(self.W2),np.zeros_like(self.b2)]
        self.v=[a.copy() for a in self.m]; self.t=0
    def forward(self,X):
        h0=relu(X@self.W0.T + self.b0); h1=relu(h0@self.W1.T + self.b1)
        z=h1@self.W2.T + self.b2; return h0,h1,z,softmax(z)
    def step_adam(self,grads,lr=1e-3,b1=0.9,b2=0.999,eps=1e-8):
        params=[self.W0,self.b0,self.W1,self.b1,self.W2,self.b2]; self.t+=1
        for i,(p,g) in enumerate(zip(params,grads)):
            self.m[i]=b1*self.m[i]+(1-b1)*g; self.v[i]=b2*self.v[i]+(1-b2)*(g*g)
            mhat=self.m[i]/(1-b1**self.t); vhat=self.v[i]/(1-b2**self.t)
            p-=lr*mhat/(np.sqrt(vhat)+eps)
    def grads(self,X,Y,l2=1e-5):
        h0,h1,z,yh=self.forward(X); N=X.shape[0]
        dz=(yh-Y)/N; gW2=dz.T@h1 + l2*self.W2; gb2=np.sum(dz,0,keepdims=True)
        dh1=dz@self.W2; dh1[h1<=0]=0
        gW1=dh1.T@h0 + l2*self.W1; gb1=np.sum(dh1,0,keepdims=True)
        dh0=dh1@self.W1; dh0[h0<=0]=0
        gW0=dh0.T@X + l2*self.W0; gb0=np.sum(dh0,0,keepdims=True)
        return [gW0,gb0,gW1,gb1,gW2,gb2], yh

def fmt(a, per=8):
    s=[]; flat=a.flatten()
    for i,v in enumerate(flat):
        s.append(f"{float(v):.8f}f")
        if (i+1)%per==0: s[-1]+="\n"
    return ", ".join(s)

def write_hdr(W0,b0,W1,b1,W2,b2,mu6,sd6,sel_idx,in_dim):
    os.makedirs(OUT_DIR,exist_ok=True)
    with open(OUT_HDR,"w") as f:
        f.write("// Auto-generated (90->30->60->4)\n#pragma once\n#include <stdint.h>\n\n")
        f.write("#define NN_INPUT_DIM 90\n#define NN_H1_DIM 30\n#define NN_H2_DIM 60\n#define NN_OUT_DIM 4\n#define NN_FEAT_DIM 6\n#define NN_WINDOW 15\n\n")
        f.write(f"static const float w0[NN_H1_DIM][NN_INPUT_DIM]={{ {fmt(W0)} }};\n")
        f.write(f"static const float b0[NN_H1_DIM]={{ {fmt(b0)} }};\n")
        f.write(f"static const float w1[NN_H2_DIM][NN_H1_DIM]={{ {fmt(W1)} }};\n")
        f.write(f"static const float b1[NN_H2_DIM]={{ {fmt(b1)} }};\n")
        f.write(f"static const float w2[NN_OUT_DIM][NN_H2_DIM]={{ {fmt(W2)} }};\n")
        f.write(f"static const float b2[NN_OUT_DIM]={{ {fmt(b2)} }};\n\n")
        f.write(f"static const float feature_mean[NN_FEAT_DIM]={{ {fmt(mu6)} }};\n")
        f.write(f"static const float feature_std[NN_FEAT_DIM] ={{ {fmt(sd6)} }};\n")
        f.write("static const uint8_t selected_feature_idx[NN_FEAT_DIM]={ %s };\n" %
                ", ".join(str(int(i)) for i in sel_idx.tolist()))

def main():
    sel_idx,mu6,sd6=load_mapping()
    X,y=load_windows(sel_idx,mu6,sd6)
    print(f"[INFO] Dataset: {X.shape[0]} samples  class counts:",
          {c:int(np.sum(y==c)) for c in range(OUT)})

    # stratified split
    Xtr,ytr,Xv,yv=split_strat(X,y,0.2)
    Ytr=one_hot(ytr,OUT); Yv=one_hot(yv,OUT)
    wrow = (lambda yy: ((np.bincount(yy,minlength=OUT).sum()/(OUT*np.maximum(1,np.bincount(yy,minlength=OUT)))))[yy])(ytr)

    rng=np.random.default_rng(SEED); mlp=MLP(rng,X.shape[1])
    best=(None,1e9)
    lr=1e-3; l2=1e-5; epochs=300
    for ep in range(epochs):
        grads, yhat_tr=mlp.grads(Xtr,Ytr,l2=l2)
        # weighted training loss (for logging only)
        tr_loss=xent(Ytr,yhat_tr,weights=wrow)
        _, _, _, yhat_v=mlp.forward(Xv)
        v_loss=xent(Yv,yhat_v); acc=float((np.argmax(yhat_v,1)==yv).mean())
        mlp.step_adam(grads,lr=lr)
        if v_loss<best[1]-1e-5: best=((mlp.W0.copy(),mlp.b0.copy(),mlp.W1.copy(),mlp.b1.copy(),mlp.W2.copy(),mlp.b2.copy()),v_loss)
        if ep%25==0 or ep==epochs-1:
            print(f"ep {ep:3d} | train {tr_loss:.4f}  val {v_loss:.4f}  acc {acc:.3f}")

    if best[0] is None:
        W0,b0,W1,b1,W2,b2=mlp.W0,mlp.b0,mlp.W1,mlp.b1,mlp.W2,mlp.b2
    else:
        W0,b0,W1,b1,W2,b2=best[0]
    write_hdr(W0,b0,W1,b1,W2,b2,mu6,sd6,sel_idx,X.shape[1])
    print(f"[INFO] Wrote {OUT_HDR}")

if __name__=="__main__":
    main()
