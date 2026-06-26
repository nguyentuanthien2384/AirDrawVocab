"""Train ONE model by name on fixed QuickDraw split. Deterministic so every
call uses the same train/val/test. Saves out/{name}_result.json + history + cm."""
import os, sys, json, time, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","1")
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score,confusion_matrix)
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier

SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception: pass

DATA_DIR="/home/claude/project/AirDrawVocab/data/npy_28"
OUT="/home/claude/experiments/out"; os.makedirs(OUT,exist_ok=True)
CATEGORIES=["apple","baseball","book","bowtie","diamond","dog","door","envelope","eye",
            "fish","hat","leaf","lightning","moon","pants","scissors","square","star","t-shirt"]
NC=len(CATEGORIES)
TRAIN_PC,VAL_PC,TEST_PC=1200,300,800
BATCH=128

def load_split():
    need=TRAIN_PC+VAL_PC+TEST_PC; rng=np.random.default_rng(SEED)
    xtr,ytr,xva,yva,xte,yte=[],[],[],[],[],[]
    for cid,cat in enumerate(CATEGORIES):
        d=np.load(f"{DATA_DIR}/{cat}.npy"); d=d[d.sum(1)>0]
        idx=rng.permutation(len(d))[:need]; d=d[idx].astype("float32")/255.0
        d=d.reshape(-1,28,28,1)
        xtr.append(d[:TRAIN_PC]); ytr.append(np.full(TRAIN_PC,cid))
        xva.append(d[TRAIN_PC:TRAIN_PC+VAL_PC]); yva.append(np.full(VAL_PC,cid))
        xte.append(d[TRAIN_PC+VAL_PC:need]); yte.append(np.full(TEST_PC,cid))
    return (np.concatenate(xtr),np.concatenate(ytr),np.concatenate(xva),
            np.concatenate(yva),np.concatenate(xte),np.concatenate(yte))

def augment(x,y):
    xp=tf.pad(x,[[0,0],[3,3],[3,3],[0,0]]); xp=tf.image.random_crop(xp,tf.shape(x)); return xp,y
def make_ds(x,y,tr):
    yc=keras.utils.to_categorical(y,NC); ds=tf.data.Dataset.from_tensor_slices((x,yc))
    if tr: return ds.shuffle(len(x),seed=SEED).batch(BATCH).map(augment,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def build_small():
    return keras.Sequential([keras.Input((28,28,1)),
        layers.Conv2D(16,3,activation="relu"),layers.MaxPooling2D(),
        layers.Conv2D(32,3,activation="relu"),layers.MaxPooling2D(),
        layers.Flatten(),layers.Dense(64,activation="relu"),layers.Dropout(0.3),
        layers.Dense(NC,activation="softmax")],name="SmallCNN")
def build_vgg():
    inp=keras.Input((28,28,1))
    x=layers.Conv2D(32,3,padding="same",activation="relu")(inp)
    x=layers.Conv2D(32,3,padding="same",activation="relu")(x); x=layers.MaxPooling2D()(x); x=layers.Dropout(0.25)(x)
    x=layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=layers.Conv2D(64,3,padding="same",activation="relu")(x); x=layers.MaxPooling2D()(x); x=layers.Dropout(0.25)(x)
    x=layers.Conv2D(128,3,padding="same",activation="relu")(x)
    x=layers.Conv2D(128,3,padding="same",activation="relu")(x); x=layers.MaxPooling2D()(x); x=layers.Dropout(0.3)(x)
    x=layers.Flatten()(x); x=layers.Dense(256,activation="relu")(x); x=layers.Dropout(0.4)(x)
    return keras.Model(inp,layers.Dense(NC,activation="softmax")(x),name="VGGStyleCNN")
def resb(x,f):
    # BN-free residual block: tránh lỗi train/val phân kỳ của BatchNorm trên Keras 3
    sc=x
    x=layers.Conv2D(f,3,padding="same",activation="relu")(x)
    x=layers.Conv2D(f,3,padding="same")(x)
    if sc.shape[-1]!=f: sc=layers.Conv2D(f,1,padding="same")(sc)
    return layers.ReLU()(layers.add([x,sc]))
def build_resnet():
    inp=keras.Input((28,28,1))
    x=layers.Conv2D(32,3,padding="same",activation="relu")(inp)
    x=resb(x,32); x=layers.MaxPooling2D()(x); x=layers.Dropout(0.2)(x)
    x=resb(x,64); x=layers.MaxPooling2D()(x); x=layers.Dropout(0.25)(x)
    x=resb(x,128); x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(128,activation="relu")(x); x=layers.Dropout(0.4)(x)
    return keras.Model(inp,layers.Dense(NC,activation="softmax")(x),name="ResNetSketch")

def top3(y,p): return float(np.mean([t in r for t,r in zip(y,np.argsort(p,1)[:,-3:])]))
def latency(m,xte):
    s=xte[:200]; m.predict(s[:8],verbose=0); t=time.time(); m.predict(s,batch_size=1,verbose=0); return (time.time()-t)/len(s)*1000

def save(name,rep,cm,hist=None):
    np.save(f"{OUT}/{name}_cm.npy",cm)
    if hist is not None: json.dump({k:[float(v) for v in vs] for k,vs in hist.items()},open(f"{OUT}/{name}_history.json","w"))
    json.dump(rep,open(f"{OUT}/{name}_result.json","w"),indent=2)
    print(f"SAVED {name}: acc={rep['test_accuracy']*100:.2f}% macroF1={rep['macro_f1']:.3f}",flush=True)

def run(name,epochs):
    xtr,ytr,xva,yva,xte,yte=load_split()
    print(f"{name}: train={len(xtr)} test={len(xte)}",flush=True)
    pc=lambda pred:{CATEGORIES[c]:float((pred[yte==c]==c).mean()) for c in range(NC)}
    if name in ("NearestCentroid","LogisticRegression"):
        clf=NearestCentroid() if name=="NearestCentroid" else SGDClassifier(loss="log_loss",max_iter=30,tol=1e-3,random_state=SEED,alpha=1e-4,early_stopping=True)
        Xtr,Xte=xtr.reshape(len(xtr),-1),xte.reshape(len(xte),-1)
        t=time.time(); clf.fit(Xtr,ytr); tt=time.time()-t
        t=time.time(); pred=clf.predict(Xte); lat=(time.time()-t)/len(Xte)*1000
        rep=dict(model=name,type="classical_baseline",params=int(Xtr.shape[1]*NC),
                 test_accuracy=float(accuracy_score(yte,pred)),top3_accuracy=None,
                 macro_f1=float(f1_score(yte,pred,average="macro")),
                 weighted_f1=float(f1_score(yte,pred,average="weighted")),
                 macro_precision=float(precision_score(yte,pred,average="macro",zero_division=0)),
                 macro_recall=float(recall_score(yte,pred,average="macro",zero_division=0)),
                 train_time_s=float(tt),latency_ms=float(lat),epochs_run=None,
                 final_train_acc=None,final_val_acc=None,per_class_acc=pc(pred))
        save(name,rep,confusion_matrix(yte,pred)); return
    builder={"SmallCNN":build_small,"VGGStyleCNN":build_vgg,"ResNetSketch":build_resnet}[name]
    keras.backend.clear_session(); gc.collect()
    m=builder(); npar=m.count_params()
    m.compile(optimizer=keras.optimizers.Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
    wpath=f"{OUT}/{name}.weights.h5"; spath=f"{OUT}/{name}_state.json"
    done_ep=0; hist_all={}
    if os.path.exists(wpath):
        m.load_weights(wpath)
        st=json.load(open(spath)); done_ep=st["epochs"]; hist_all=st.get("history",{})
        print(f"resume {name} from {done_ep} epochs",flush=True)
    cbs=[keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6)]
    t=time.time(); h=m.fit(make_ds(xtr,ytr,True),validation_data=make_ds(xva,yva,False),
                            epochs=epochs,callbacks=cbs,verbose=2); tt=time.time()-t
    for k,vs in h.history.items(): hist_all.setdefault(k,[]).extend([float(v) for v in vs])
    done_ep+=len(h.history["loss"])
    m.save_weights(wpath)
    json.dump({"epochs":done_ep,"history":hist_all},open(spath,"w"))
    p=m.predict(make_ds(xte,yte,False),verbose=0); pred=p.argmax(1)
    rep=dict(model=name,type="deep_learning",params=int(npar),
             test_accuracy=float(accuracy_score(yte,pred)),top3_accuracy=top3(yte,p),
             macro_f1=float(f1_score(yte,pred,average="macro")),
             weighted_f1=float(f1_score(yte,pred,average="weighted")),
             macro_precision=float(precision_score(yte,pred,average="macro",zero_division=0)),
             macro_recall=float(recall_score(yte,pred,average="macro",zero_division=0)),
             train_time_s=float(tt),latency_ms=float(latency(m,xte)),epochs_run=done_ep,
             final_train_acc=float(hist_all["accuracy"][-1]),final_val_acc=float(hist_all["val_accuracy"][-1]),
             per_class_acc=pc(pred))
    save(name,rep,confusion_matrix(yte,pred),hist_all)

if __name__=="__main__":
    name=sys.argv[1]; epochs=int(sys.argv[2]) if len(sys.argv)>2 else 16
    run(name,epochs)
