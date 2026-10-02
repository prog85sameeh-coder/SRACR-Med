from __future__ import print_function
import argparse
import io
import json
import math
import os
import random
import zlib
import gzip
import pickle
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DATASETS = {
    "chest": {
        "source": "Detection of Pneumonia Disease using Chest Radiogr",
        "model": "chest_xray_resnet18.pt",
        "cache": "chest_all_seed42_chest_xray_resnet18.pkl.gz",
    },
    "retinal": {
        "source": "Retinal Dataset",
        "model": "retinal_11class_resnet18.pt",
        "cache": "retinal_all_seed42_retinal_11class_resnet18.pkl.gz",
    },
    "split": {
        "source": "split",
        "model": "fundus_4class_resnet18.pt",
        "cache": "split_all_seed42_fundus_4class_resnet18.pkl.gz",
    },
}

CLINICAL = {
    "NORMAL_Images": 0.35,
    "Chest_Radiograh Screeing (X-Ray)": 0.75,
    "normal": 0.35,
    "cataract": 0.65,
    "glaucoma": 0.85,
    "diabetic_retinopathy": 0.85,
    "No Age-Related Macular Degeneration (Healthy regarding AMD)": 0.35,
    "Mild Glaucoma": 0.65,
    "Moderate Glaucoma": 0.80,
    "Severe Glaucoma": 0.95,
    "AdvancedEnd-stage Glaucoma": 1.00,
    "Mild Diabetic Retinopathy": 0.65,
    "Moderate Diabetic Retinopathy": 0.80,
    "Severe Diabetic Retinopathy": 0.95,
    "Proliferative Diabetic Retinopathy (PDR)": 1.00,
    "Dry Age-Related Macular Degeneration": 0.80,
    "Wet Age-Related Macular Degeneration": 0.95,
}

WEIGHTS = (0.30, 0.25, 0.30, 0.15)
THRESHOLDS = (0.30, 0.50, 0.70)
HYSTERESIS = 0.05
COVERAGE = {1:0.25, 2:0.50, 3:0.75, 4:1.00}
THREATS = (0.10, 0.40, 0.70, 0.95)
SEQUENCE_SEEDS = {
    0.10: (1742, 2751, 3760, 4769, 5778),
    0.40: (6842, 7851, 8860, 9869, 10878),
    0.70: (11942, 12951, 13960, 14969, 15978),
    0.95: (16192, 17201, 18210, 19219, 20228),
}

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


class GradCAM(object):
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if target_index is None:
            target_index = int(probs.argmax(dim=1).item())
        logits[:, target_index].sum().backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0,0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach(), target_index, probs.detach()[0]

    def remove(self):
        self._fh.remove(); self._bh.remove()


def normalize_tensor(x):
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1,3,1,1)
    std = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1,3,1,1)
    return (x - mean) / std


def raw_tensor(image, size, device):
    tf = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    return tf(image).unsqueeze(0).to(device)


def normalized_entropy(probs):
    p = probs.detach().float().clamp_min(1e-8)
    p = p / p.sum().clamp_min(1e-8)
    h = -(p * p.log()).sum().item()
    return float(np.clip(h / math.log(max(int(p.numel()), 2)), 0.0, 1.0))


def perturb_raw(x, idx, noise_std=0.02, brightness_delta=0.05, contrast_delta=0.05):
    y = x.clone()
    mode = idx % 4
    if mode == 0:
        y = y + torch.randn_like(y) * noise_std
    elif mode == 1:
        y = y + brightness_delta
    elif mode == 2:
        mean = y.mean(dim=(2,3), keepdim=True)
        y = (y - mean) * (1.0 + contrast_delta) + mean
    else:
        y = torch.roll(y, shifts=(2,-2), dims=(2,3))
    return y.clamp(0.0, 1.0)


def cosine_map_similarity(a,b):
    av=a.reshape(-1).float(); bv=b.reshape(-1).float()
    if float(av.norm()) < 1e-10 or float(bv.norm()) < 1e-10:
        return 0.0
    return float(F.cosine_similarity(av.unsqueeze(0), bv.unsqueeze(0)).item())


def saliency_stability(model, cam_engine, raw_x, base_cam, n_perturb=4):
    with torch.no_grad():
        target = int(model(normalize_tensor(raw_x)).argmax(1).item())
    sims=[]
    for i in range(int(n_perturb)):
        xp_raw = perturb_raw(raw_x, i)
        cam_p, _, _ = cam_engine(normalize_tensor(xp_raw), target_index=target)
        sims.append(cosine_map_similarity(base_cam, cam_p))
    return float(np.clip(np.mean(sims),0.0,1.0)) if sims else 1.0


def saliency_burden(reliable):
    s=np.asarray(reliable,dtype=np.float32)
    return float(np.clip(0.5*float(s.mean()) + 0.5*float(np.percentile(s,95)), 0.0, 1.0))


def tile_scores(map2d, image_hw, tile_size=64):
    h,w=image_hw
    arr=np.asarray(map2d,dtype=np.float32)
    im=Image.fromarray(np.uint8(np.clip(arr,0,1)*255)).resize((w,h), RESAMPLE_BILINEAR)
    r=np.asarray(im,dtype=np.float32)/255.0
    scores=[]; idx=0
    for y0 in range(0,h,tile_size):
        y1=min(h,y0+tile_size)
        for x0 in range(0,w,tile_size):
            x1=min(w,x0+tile_size)
            scores.append((idx,float(r[y0:y1,x0:x1].mean())))
            idx += 1
    return scores


def select_top(scores, fraction):
    n=len(scores)
    k=max(1,min(n,int(np.ceil(float(fraction)*n))))
    ranked=sorted(scores,key=lambda z:z[1],reverse=True)
    return set(idx for idx,_ in ranked[:k])


def rankdata_average(x):
    s=pd.Series(np.asarray(x,dtype=float))
    return s.rank(method="average").to_numpy(dtype=float)


def spearman_like(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if len(a)!=len(b) or len(a)<2:
        return np.nan
    ra=rankdata_average(a); rb=rankdata_average(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra,rb)[0,1])


def jaccard(a,b):
    u=set(a).union(set(b)); i=set(a).intersection(set(b))
    return 1.0 if not u else float(len(i))/float(len(u))


def load_checkpoint(path, device):
    ckpt=torch.load(str(path),map_location=device)
    state=ckpt.get("model_state_dict") or ckpt.get("state_dict")
    classes=ckpt.get("class_names") or ckpt.get("classes")
    image_size=int(ckpt.get("image_size",224))
    if state is None:
        raise ValueError("Unsupported checkpoint format: %s" % path)
    if not classes:
        n=int(state["fc.weight"].shape[0]); classes=["class_%d"%i for i in range(n)]
    model=models.resnet18(weights=None)
    model.fc=nn.Linear(model.fc.in_features,len(classes))
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model,list(classes),image_size


def canonical_key(path):
    s=str(path).replace("\\","/")
    low=s.lower()
    marker="/sracr-med/"
    pos=low.find(marker)
    return low[pos+len(marker):] if pos>=0 else low


def remap_path(original_path, dataset_root):
    p=Path(str(original_path))
    if p.exists():
        return p
    wp=PureWindowsPath(str(original_path))
    parts=list(wp.parts)
    idx=None
    for i,x in enumerate(parts):
        if str(x).lower()=="sracr-med":
            idx=i; break
    if idx is None:
        raise FileNotFoundError("Cannot remap path: %s" % original_path)
    return Path(dataset_root).joinpath(*parts[idx+1:])


def apply_shift(image, name):
    if name=="clean":
        return image.copy()
    if name=="gaussian_blur_r1p5":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name=="jpeg_q40":
        bio=io.BytesIO()
        image.save(bio, format="JPEG", quality=40, subsampling=2, optimize=False, progressive=False)
        bio.seek(0)
        with Image.open(bio) as im:
            return im.convert("RGB").copy()
    if name=="downsample50_bilinear":
        w,h=image.size
        small=image.resize((max(1,w//2),max(1,h//2)), RESAMPLE_BILINEAR)
        return small.resize((w,h), RESAMPLE_BILINEAR)
    raise ValueError(name)


def clinical_score(name):
    if name in CLINICAL:
        return float(CLINICAL[name])
    lower={str(k).lower():v for k,v in CLINICAL.items()}
    return float(lower.get(str(name).lower(),0.65))


def compute_features(model, cam_engine, image, classes, image_size, device, deterministic_seed):
    random.seed(int(deterministic_seed))
    np.random.seed(int(deterministic_seed) % (2**32-1))
    torch.manual_seed(int(deterministic_seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(deterministic_seed))
    raw=raw_tensor(image,image_size,device)
    x=normalize_tensor(raw)
    with torch.no_grad():
        probs=torch.softmax(model(x),dim=1)[0]
        pred=int(probs.argmax().item())
        unc=normalized_entropy(probs)
    cam,_,_=cam_engine(x,target_index=pred)
    # reset seed so Gaussian component of reliability is deterministic and image/shift-specific
    torch.manual_seed(int(deterministic_seed)+101)
    rel=saliency_stability(model,cam_engine,raw,cam,n_perturb=4)
    reliable=np.clip(cam.cpu().numpy().astype(np.float32)*rel,0,1)
    return {
        "pred_idx":pred,
        "pred_class":classes[pred],
        "probs":probs.cpu().numpy().astype(np.float32),
        "confidence":float(probs[pred].item()),
        "uncertainty":float(unc),
        "reliability":float(rel),
        "reliable":reliable,
        "burden":saliency_burden(reliable),
    }


class Controller(object):
    def __init__(self):
        self.level=1
    def desired(self,risk,threat):
        if threat>=0.90 or risk>=THRESHOLDS[2]: return 4
        if risk>=THRESHOLDS[1]: return 3
        if risk>=THRESHOLDS[0]: return 2
        return 1
    def update(self,risk,threat):
        d=self.desired(risk,threat); old=self.level
        if d>old:
            self.level=d
        elif d<old:
            boundary=THRESHOLDS[max(0,old-2)] if old>1 else 0.0
            if float(risk)<boundary-HYSTERESIS and float(threat)<0.90:
                self.level=max(d,old-1)
        return self.level


def global_risk(burden,clinical,threat,uncertainty):
    return float(np.clip(WEIGHTS[0]*burden + WEIGHTS[1]*clinical + WEIGHTS[2]*threat + WEIGHTS[3]*uncertainty,0,1))


def actual_fraction(n_tiles, coverage):
    k=max(1,min(n_tiles,int(np.ceil(float(coverage)*n_tiles))))
    return float(k)/float(n_tiles)


def bootstrap_mean_ci(values, reps, seed):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)==0: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(int(seed))
    means=[]
    for _ in range(int(reps)):
        means.append(float(np.mean(x[rng.integers(0,len(x),len(x))])))
    return float(np.mean(x)), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))


def summarize_image_level(df, bootstrap_reps, seed):
    rows=[]
    for (dataset,shift),g in df.groupby(["dataset","shift"],sort=False):
        y_true=g["true_label"].astype(str).to_numpy()
        y_pred=g["shift_predicted_class"].astype(str).to_numpy()
        acc=float(np.mean(g["correct"].to_numpy(dtype=float)))
        f1=float(f1_score(y_true,y_pred,average="macro",zero_division=0))
        bal=float(balanced_accuracy_score(y_true,y_pred))
        acc_m,acc_lo,acc_hi=bootstrap_mean_ci(g["correct"],bootstrap_reps,seed+11)
        row={"dataset":dataset,"shift":shift,"n_images":len(g),"accuracy":acc,"accuracy_ci95_low":acc_lo,"accuracy_ci95_high":acc_hi,"macro_f1":f1,"balanced_accuracy":bal}
        for col in ["prediction_agreement_clean","uncertainty","reliability","tile_spearman_vs_clean","top50_jaccard_vs_clean","burden"]:
            m,lo,hi=bootstrap_mean_ci(g[col],bootstrap_reps,seed+abs(zlib.crc32((dataset+shift+col).encode()))%100000)
            row[col+"_mean"]=m; row[col+"_ci95_low"]=lo; row[col+"_ci95_high"]=hi
        rows.append(row)
    return pd.DataFrame(rows)


def controller_records(dataset, condition_records, clean_records):
    rows=[]
    n=len(condition_records)
    for threat in THREATS:
        for seq_seed in SEQUENCE_SEEDS[float(threat)]:
            order=np.random.default_rng(int(seq_seed)).permutation(n)
            c_shift=Controller(); c_clean=Controller()
            for pos in order:
                i=int(pos); sr=condition_records[i]; cr=clean_records[i]
                risk_s=global_risk(sr["burden"],sr["clinical"],threat,sr["uncertainty"])
                risk_c=global_risk(cr["burden"],cr["clinical"],threat,cr["uncertainty"])
                lev_s=c_shift.update(risk_s,threat); lev_c=c_clean.update(risk_c,threat)
                rows.append({
                    "dataset":dataset,"shift":sr["shift"],"image_index":i,"sequence_seed":seq_seed,"threat":threat,
                    "global_risk":risk_s,"clean_global_risk":risk_c,"risk_delta":risk_s-risk_c,
                    "controller_level":lev_s,"clean_controller_level":lev_c,
                    "level_agreement_clean":int(lev_s==lev_c),"abs_level_delta":abs(lev_s-lev_c),
                    "protected_fraction":actual_fraction(sr["n_tiles"],COVERAGE[lev_s]),
                    "clean_protected_fraction":actual_fraction(cr["n_tiles"],COVERAGE[lev_c]),
                })
    return rows


def summarize_controller(detail, bootstrap_reps, seed):
    # average repetitions per image first, then bootstrap images
    g=detail.groupby(["dataset","shift","threat","image_index"],as_index=False).agg({
        "global_risk":"mean","clean_global_risk":"mean","risk_delta":"mean",
        "controller_level":"mean","clean_controller_level":"mean","level_agreement_clean":"mean",
        "abs_level_delta":"mean","protected_fraction":"mean","clean_protected_fraction":"mean"
    })
    rows=[]
    for (dataset,shift,threat),x in g.groupby(["dataset","shift","threat"],sort=False):
        row={"dataset":dataset,"shift":shift,"threat":threat,"n_images":len(x)}
        for col in ["global_risk","risk_delta","controller_level","level_agreement_clean","abs_level_delta","protected_fraction"]:
            m,lo,hi=bootstrap_mean_ci(x[col],bootstrap_reps,seed+abs(zlib.crc32((dataset+shift+str(threat)+col).encode()))%100000)
            row[col+"_mean"]=m; row[col+"_ci95_low"]=lo; row[col+"_ci95_high"]=hi
        rows.append(row)
    return pd.DataFrame(rows),g


def cross_dataset_macro(summary, controller_summary):
    s=summary.groupby("shift",as_index=False).agg({
        "accuracy":"mean","macro_f1":"mean","balanced_accuracy":"mean",
        "prediction_agreement_clean_mean":"mean","uncertainty_mean":"mean","reliability_mean":"mean",
        "tile_spearman_vs_clean_mean":"mean","top50_jaccard_vs_clean_mean":"mean","burden_mean":"mean"
    })
    c=controller_summary.groupby(["shift","threat"],as_index=False).agg({
        "global_risk_mean":"mean","risk_delta_mean":"mean","controller_level_mean":"mean",
        "level_agreement_clean_mean":"mean","abs_level_delta_mean":"mean","protected_fraction_mean":"mean"
    })
    return s,c


def run(args):
    seed=int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device="cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    root=Path(args.dataset_root)
    model_dir=Path(args.primary_model_dir)
    cache_dir=Path(args.feature_cache_dir)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    shifts=["clean","gaussian_blur_r1p5","jpeg_q40","downsample50_bilinear"]
    aliases=list(DATASETS.keys()) if args.source=="all" else [args.source]
    all_image=[]; all_controller=[]

    for alias in aliases:
        info=DATASETS[alias]
        print("\n=== %s ===" % alias, flush=True)
        model,classes,image_size=load_checkpoint(model_dir/info["model"],device)
        with gzip.open(cache_dir/info["cache"],"rb") as f:
            payload=pickle.load(f)
        clean_cache=payload["records"]
        cam_engine=GradCAM(model,model.layer4[-1].conv2)
        shifted_by_name={s:[] for s in shifts if s!="clean"}
        clean_for_controller=[]
        try:
            for i,cr in enumerate(clean_cache):
                path=remap_path(cr["image_path"],root)
                with Image.open(path) as im:
                    image=im.convert("RGB")
                n_tiles=len(cr["tile_ids"])
                clean_pairs=list(zip([int(x) for x in cr["tile_ids"]],[float(x) for x in cr["tile_scores"]]))
                clean_top50=select_top(clean_pairs,0.50)
                clean_for_controller.append({
                    "shift":"clean","burden":float(cr["reliable_saliency_burden"]),"clinical":float(cr["clinical"]),
                    "uncertainty":float(cr["uncertainty"]),"n_tiles":n_tiles
                })
                all_image.append({
                    "dataset":alias,"image_index":i,"shift":"clean","image_path":str(path),"true_label":str(cr["true_label"]),
                    "clean_predicted_class":str(cr["predicted_class"]),"shift_predicted_class":str(cr["predicted_class"]),
                    "correct":int(str(cr["predicted_class"])==str(cr["true_label"])),"prediction_agreement_clean":1,
                    "uncertainty":float(cr["uncertainty"]),"uncertainty_delta":0.0,
                    "reliability":float(cr["reliability"]),"reliability_delta":0.0,
                    "burden":float(cr["reliable_saliency_burden"]),"burden_delta":0.0,
                    "tile_spearman_vs_clean":1.0,"top50_jaccard_vs_clean":1.0,"n_tiles":n_tiles
                })
                for sidx,shift in enumerate(shifts[1:],start=1):
                    shifted=apply_shift(image,shift)
                    det_seed=seed + (i+1)*1009 + sidx*100003 + (abs(zlib.crc32(alias.encode()))%10000)
                    feat=compute_features(model,cam_engine,shifted,classes,image_size,device,det_seed)
                    scores=tile_scores(feat["reliable"],(cr["height"],cr["width"]),64)
                    vals=np.asarray([v for _,v in scores],dtype=float)
                    clean_vals=np.asarray([v for _,v in clean_pairs],dtype=float)
                    top50=select_top(scores,0.50)
                    clinical=clinical_score(feat["pred_class"])
                    rec={"shift":shift,"burden":feat["burden"],"clinical":clinical,"uncertainty":feat["uncertainty"],"n_tiles":n_tiles}
                    shifted_by_name[shift].append(rec)
                    all_image.append({
                        "dataset":alias,"image_index":i,"shift":shift,"image_path":str(path),"true_label":str(cr["true_label"]),
                        "clean_predicted_class":str(cr["predicted_class"]),"shift_predicted_class":feat["pred_class"],
                        "correct":int(feat["pred_class"]==str(cr["true_label"])),
                        "prediction_agreement_clean":int(feat["pred_class"]==str(cr["predicted_class"])),
                        "uncertainty":feat["uncertainty"],"uncertainty_delta":feat["uncertainty"]-float(cr["uncertainty"]),
                        "reliability":feat["reliability"],"reliability_delta":feat["reliability"]-float(cr["reliability"]),
                        "burden":feat["burden"],"burden_delta":feat["burden"]-float(cr["reliable_saliency_burden"]),
                        "tile_spearman_vs_clean":spearman_like(clean_vals,vals),
                        "top50_jaccard_vs_clean":jaccard(clean_top50,top50),"n_tiles":n_tiles
                    })
                if (i+1)%25==0 or i+1==len(clean_cache):
                    print("  processed %d/%d"%(i+1,len(clean_cache)),flush=True)
        finally:
            cam_engine.remove()

        # controller comparisons: clean vs each shift; include clean self-comparison once
        all_controller.extend(controller_records(alias, clean_for_controller, clean_for_controller))
        for shift in shifts[1:]:
            all_controller.extend(controller_records(alias, shifted_by_name[shift], clean_for_controller))

    image_df=pd.DataFrame(all_image)
    ctrl_detail=pd.DataFrame(all_controller)
    image_summary=summarize_image_level(image_df,args.bootstrap,seed)
    ctrl_summary,ctrl_image=summarize_controller(ctrl_detail,args.bootstrap,seed)
    macro_img,macro_ctrl=cross_dataset_macro(image_summary,ctrl_summary)

    # monotonic controller check across threat for every dataset/shift
    mono=[]
    for (dataset,shift),g in ctrl_summary.groupby(["dataset","shift"]):
        gg=g.sort_values("threat")
        vals=gg["protected_fraction_mean"].to_numpy(dtype=float)
        mono.append({"dataset":dataset,"shift":shift,"monotonic_non_decreasing":bool(np.all(np.diff(vals)>=-1e-12)),
                     "protected_fraction_T0p10":float(gg.iloc[0]["protected_fraction_mean"]),
                     "protected_fraction_T0p95":float(gg.iloc[-1]["protected_fraction_mean"])})
    mono_df=pd.DataFrame(mono)

    files={
        "R2C4_ALL_DATASETS_controlled_shift_image_level.csv":image_df,
        "R2C4_ALL_DATASETS_controlled_shift_summary.csv":image_summary,
        "R2C4_ALL_DATASETS_controlled_shift_controller_detail.csv":ctrl_detail,
        "R2C4_ALL_DATASETS_controlled_shift_controller_image_level.csv":ctrl_image,
        "R2C4_ALL_DATASETS_controlled_shift_controller_summary.csv":ctrl_summary,
        "R2C4_CROSS_DATASET_MACRO_shift_summary.csv":macro_img,
        "R2C4_CROSS_DATASET_MACRO_controller_summary.csv":macro_ctrl,
        "R2C4_controller_monotonicity_checks.csv":mono_df,
    }
    for name,df in files.items():
        df.to_csv(out/name,index=False,encoding="utf-8-sig")
        print("saved",out/name)

    config={
        "seed":seed,"device":device,"shifts":shifts,
        "shift_definitions":{
            "gaussian_blur_r1p5":"PIL GaussianBlur radius=1.5",
            "jpeg_q40":"JPEG encode/decode quality=40, subsampling=2",
            "downsample50_bilinear":"resize to 50% native width/height then restore with bilinear interpolation"
        },
        "note":"These controlled shifts are distinct from the four saliency-reliability perturbations used inside Q_S (noise, brightness, contrast, 2-pixel roll).",
        "saliency":{"target_layer":"ResNet18 layer4[-1].conv2","target_rule":"shifted-image argmax, fixed across the four reliability perturbations","reliability_perturbations":4},
        "controller":{"weights":WEIGHTS,"thresholds":THRESHOLDS,"hysteresis":HYSTERESIS,"coverage":COVERAGE,"threats":THREATS,"sequence_seeds":SEQUENCE_SEEDS},
        "primary_outcomes":["diagnostic accuracy/macro-F1","prediction agreement with clean","uncertainty","saliency reliability","tile-score Spearman correlation vs clean","top-50% tile Jaccard vs clean","controller-level agreement vs clean","protected-fraction response across T"],
        "scope":"controlled distribution-shift stress test on internal held-out images; not an external-site or multicenter validation and not patient-level validation"
    }
    with open(out/"R2C4_controlled_shift_protocol.json","w",encoding="utf-8") as f:
        json.dump(config,f,ensure_ascii=False,indent=2)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",choices=["all","chest","retinal","split"],default="all")
    ap.add_argument("--cpu",action="store_true")
    ap.add_argument("--dataset-root",required=True)
    ap.add_argument("--primary-model-dir",required=True)
    ap.add_argument("--feature-cache-dir",default="feature_cache")
    ap.add_argument("--output-dir",default="outputs_r2c4")
    ap.add_argument("--bootstrap",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=42)
    run(ap.parse_args())
