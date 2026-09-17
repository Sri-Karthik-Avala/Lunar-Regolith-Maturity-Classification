# made by - Karthik

import os
import sys
import math
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision


SEED = 1234
FOLDS = 5
EPOCHS = 16
BATCH = 24
RESIZE = 480
CROP = 448
LR_BB = 2e-4
LR_HEAD = 1e-3
WD = 1e-4
DROP = 0.2
W_CLS = 1.0
REG_COLS = ["nanophase_iron_darkening_anomaly", "agglutinate_glass_anomaly",
            "reflectance_anomaly_score", "mean_grain_size_microns"]
REG_LOSS_W = np.array([0.3, 0.3, 1.0, 2.0], dtype=np.float32)
GRAIN_IDX = 3
LOW = np.array([-1.0, -1.0, -1.0, 40.0], dtype=np.float32)
HIGH = np.array([1.0, 1.0, 1.0, 800.0], dtype=np.float32)
NUM_CLASSES = 4
NFEAT = 13


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def find_root():
    env = os.environ.get("REGOLITH_ROOT")
    cand = []
    if env:
        cand.append(env)
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for base in [os.getcwd(), here]:
        cand += [base,
                 os.path.join(base, "public"),
                 os.path.join(base, "dataset", "public"),
                 os.path.join(base, "dataset"),
                 os.path.join(base, "data"),
                 os.path.join(base, ".."),
                 os.path.join(base, "..", "public")]
    seen = set()
    for c in cand:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        tr = os.path.join(c, "train.csv")
        te = os.path.join(c, "test.csv")
        if os.path.isfile(tr) and os.path.isfile(te):
            df = pd.read_csv(tr, nrows=1)
            if os.path.isfile(os.path.join(c, df["image"].iloc[0])):
                return c
    raise FileNotFoundError("could not locate train.csv/test.csv with images")


def stratified_folds(labels, n_folds, seed):
    labels = np.asarray(labels)
    fold = np.full(len(labels), -1, dtype=np.int64)
    rng = np.random.RandomState(seed)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            fold[j] = i % n_folds
    return fold


def macro_f1(y_true, y_pred, k=NUM_CLASSES):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(k):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def tex_feats(a):
    g = a.mean(2)
    R = a[:, :, 0].mean()
    B = a[:, :, 2].mean()
    mask = g > 0.02
    fg = g[mask] if mask.sum() > 10 else g.ravel()
    gm = fg.mean()
    gsd = fg.std()
    gy, gx = np.gradient(g)
    gradmag = np.sqrt(gx * gx + gy * gy).mean()
    lap = (g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2] - 4 * g[1:-1, 1:-1])
    lapvar = lap.var()
    G2 = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = G2.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()
    tot = G2.sum() + 1e-9
    midE = G2[(r >= 0.05 * rmax) & (r < 0.2 * rmax)].sum() / tot
    highE = G2[r >= 0.2 * rmax].sum() / tot
    centroid = (r * G2).sum() / tot / rmax
    bright = (g > 0.6).mean()
    dark = (g < 0.15).mean()
    p90 = np.percentile(g, 90)
    sat = (a.max(2) - a.min(2)).mean()
    fgf = mask.mean()
    return np.array([gm, gsd, gradmag, lapvar, midE, highE, centroid,
                     bright, dark, p90, sat, R - B, fgf], dtype=np.float32)


def load_cache_and_feats(root, paths, size):
    cache = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    feats = np.zeros((len(paths), NFEAT), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(os.path.join(root, p)).convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255.0
        feats[i] = tex_feats(a)
        cache[i] = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return cache, feats


class RegDataset(Dataset):
    def __init__(self, cache, feats, targets, train, crop):
        self.cache = cache
        self.feats = feats
        self.targets = targets
        self.train = train
        self.crop = crop
        self.full = cache.shape[1]

    def __len__(self):
        return self.cache.shape[0]

    def _aug(self, img):
        s = self.full - self.crop
        if self.train and s > 0:
            oy = random.randint(0, s)
            ox = random.randint(0, s)
        else:
            oy = ox = s // 2
        img = img[oy:oy + self.crop, ox:ox + self.crop]
        if self.train:
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k)
            if random.random() < 0.5:
                img = img[:, ::-1]
            if random.random() < 0.5:
                img = img[::-1, :]
        return np.ascontiguousarray(img)

    def __getitem__(self, i):
        img = self._aug(self.cache[i])
        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        fe = torch.from_numpy(self.feats[i])
        if self.targets is None:
            return x, fe, 0
        return x, fe, torch.from_numpy(self.targets[i])


def load_backbone():
    models = torchvision.models
    try:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        print("backbone: efficientnet_b0 IMAGENET1K_V1 via weights enum")
    except Exception:
        try:
            m = models.efficientnet_b0(pretrained=True)
            print("backbone: efficientnet_b0 via pretrained=True")
        except Exception:
            m = models.efficientnet_b0(weights=None)
            found = glob.glob(os.path.expanduser("~/.cache/torch/hub/checkpoints/efficientnet_b0_*.pth"))
            if found:
                m.load_state_dict(torch.load(found[0], map_location="cpu"))
                print("backbone: efficientnet_b0 from local cache " + os.path.basename(found[0]))
            else:
                print("WARNING: efficientnet_b0 pretrained weights unavailable -> random init")
    return m


class RegolithNet(nn.Module):
    def __init__(self, n_feat=NFEAT, drop=DROP):
        super().__init__()
        bb = load_backbone()
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        bb_out = bb.classifier[1].in_features
        self.aux = nn.Sequential(nn.Linear(n_feat, 64), nn.ReLU(inplace=True),
                                 nn.Linear(64, 64), nn.ReLU(inplace=True))
        fused = bb_out + 64
        self.drop = nn.Dropout(drop)
        self.cls_head = nn.Sequential(nn.Linear(fused, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(drop), nn.Linear(256, NUM_CLASSES))
        self.reg_head = nn.Sequential(nn.Linear(fused, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(drop), nn.Linear(256, 4))

    def forward(self, x, feats):
        f = self.pool(self.features(x)).flatten(1)
        a = self.aux(feats)
        z = self.drop(torch.cat([f, a], 1))
        return self.cls_head(z), self.reg_head(z)

    def param_groups(self, lr_bb, lr_head):
        bb = list(self.features.parameters())
        head = (list(self.aux.parameters()) + list(self.cls_head.parameters()) +
                list(self.reg_head.parameters()))
        return [{"params": bb, "lr": lr_bb}, {"params": head, "lr": lr_head}]


def d4_views(x):
    outs = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        outs.append(r)
        outs.append(torch.flip(r, dims=(3,)))
    return outs


def infer_tta(model, cache, feats, fmean, fstd, pm, ps, device, bs=32):
    model.eval()
    fz = (feats - fmean) / fstd
    ds = RegDataset(cache, fz, None, train=False, crop=CROP)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    probs = np.zeros((len(ds), NUM_CLASSES), dtype=np.float64)
    regs = np.zeros((len(ds), 4), dtype=np.float64)
    m = torch.tensor(pm, device=device).view(1, 3, 1, 1)
    sd = torch.tensor(ps, device=device).view(1, 3, 1, 1)
    pos = 0
    with torch.no_grad():
        for x, fe, _ in dl:
            x = x.to(device, non_blocking=True)
            fe = fe.to(device, non_blocking=True)
            x = (x - m) / sd
            pa = torch.zeros(x.size(0), NUM_CLASSES, device=device)
            ra = torch.zeros(x.size(0), 4, device=device)
            for v in d4_views(x):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    cl, rg = model(v, fe)
                pa += F.softmax(cl.float(), dim=1)
                ra += rg.float()
            pa /= 8.0
            ra /= 8.0
            n = x.size(0)
            probs[pos:pos + n] = pa.cpu().numpy()
            regs[pos:pos + n] = ra.cpu().numpy()
            pos += n
    return probs, regs


def train_fold(tr_cache, tr_feats, tr_tgt, va_cache, va_feats, fmean, fstd, pm, ps, device):
    set_seed(SEED)
    model = RegolithNet().to(device)
    opt = torch.optim.AdamW(model.param_groups(LR_BB, LR_HEAD), weight_decay=WD)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    fz = (tr_feats - fmean) / fstd
    ds = RegDataset(tr_cache, fz, tr_tgt, train=True, crop=CROP)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    steps = len(dl)
    m = torch.tensor(pm, device=device).view(1, 3, 1, 1)
    sd = torch.tensor(ps, device=device).view(1, 3, 1, 1)
    rw = torch.tensor(REG_LOSS_W, device=device)
    base = [LR_BB, LR_HEAD]
    for ep in range(EPOCHS):
        model.train()
        for it, (x, fe, t) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            fe = fe.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            x = (x - m) / sd
            lab = t[:, 0].long()
            reg_t = t[:, 1:5]
            prog = (ep + it / steps) / EPOCHS
            f = 0.5 * (1 + math.cos(math.pi * prog))
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = base[gi] * f
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                cl, rg = model(x, fe)
                lc = F.cross_entropy(cl, lab, label_smoothing=0.05)
                lr_reg = F.smooth_l1_loss(rg, reg_t, reduction="none").mean(0)
                lr_reg = (lr_reg * rw).sum() / rw.sum()
                loss = W_CLS * lc + lr_reg
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
    va_probs, va_regs = infer_tta(model, va_cache, va_feats, fmean, fstd, pm, ps, device)
    return model, va_probs, va_regs


def calibrate_class(oof_probs, oof_grain, y_true, grain_mu, grain_sd):
    logp_head = np.log(np.clip(oof_probs, 1e-9, 1.0))
    d = oof_grain[:, None] - grain_mu[None, :]
    logp_grain = -0.5 * (d / grain_sd[None, :]) ** 2 - np.log(grain_sd[None, :])
    logp_grain = logp_grain - logp_grain.max(1, keepdims=True)
    best_lam, best_f1 = 0.0, -1.0
    for lam in np.linspace(0, 1, 21):
        pred = ((1 - lam) * logp_head + lam * logp_grain).argmax(1)
        f = macro_f1(y_true, pred)
        if f > best_f1:
            best_f1, best_lam = f, lam
    return best_lam, best_f1


def apply_class(probs, grain, lam, grain_mu, grain_sd):
    logp_head = np.log(np.clip(probs, 1e-9, 1.0))
    d = grain[:, None] - grain_mu[None, :]
    logp_grain = -0.5 * (d / grain_sd[None, :]) ** 2 - np.log(grain_sd[None, :])
    logp_grain = logp_grain - logp_grain.max(1, keepdims=True)
    return ((1 - lam) * logp_head + lam * logp_grain).argmax(1)


def calibrate_reg_col(cnn, cmean_by_pred, y_true):
    best = None
    for w in np.linspace(0, 1, 21):
        blended = w * cnn + (1 - w) * cmean_by_pred
        A = np.vstack([blended, np.ones_like(blended)]).T
        ab, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
        rmse = np.sqrt(np.mean((A @ ab - y_true) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, w, ab[0], ab[1])
    return best[1], best[2], best[3]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    set_seed(SEED)
    root = find_root()
    tr = pd.read_csv(os.path.join(root, "train.csv"))
    te = pd.read_csv(os.path.join(root, "test.csv"))
    labels = tr["label"].values.astype(np.int64)
    reg_raw = tr[REG_COLS].values.astype(np.float32)

    reg_mean = reg_raw.mean(0)
    reg_std = reg_raw.std(0) + 1e-6
    reg_z = (reg_raw - reg_mean) / reg_std
    tgt = np.concatenate([labels[:, None].astype(np.float32), reg_z], axis=1)

    grain_mu = np.array([reg_raw[labels == c, GRAIN_IDX].mean() for c in range(NUM_CLASSES)])
    grain_sd = np.array([reg_raw[labels == c, GRAIN_IDX].std() + 1e-6 for c in range(NUM_CLASSES)])
    cmean = np.array([[reg_raw[labels == c, j].mean() for j in range(4)] for c in range(NUM_CLASSES)])

    tr_cache, tr_feats = load_cache_and_feats(root, tr["image"].values, RESIZE)
    te_cache, te_feats = load_cache_and_feats(root, te["image"].values, RESIZE)
    fmean = tr_feats.mean(0)
    fstd = tr_feats.std(0) + 1e-6
    sidx = np.random.RandomState(0).choice(len(tr_cache), size=min(300, len(tr_cache)), replace=False)
    px = tr_cache[sidx][:, ::4, ::4, :].reshape(-1, 3).astype(np.float32) / 255.0
    pm = px.mean(0)
    ps = px.std(0) + 1e-6

    fold = stratified_folds(labels, FOLDS, SEED)
    N = len(tr)
    oof_probs = np.zeros((N, NUM_CLASSES))
    oof_regs = np.zeros((N, 4))
    test_probs = np.zeros((len(te), NUM_CLASSES))
    test_regs = np.zeros((len(te), 4))
    oof_mask = np.zeros(N, dtype=bool)

    agg_only = bool(os.environ.get("REGOLITH_AGG_ONLY"))
    ckdir = os.path.join("working", "_ckpt")
    os.makedirs(ckdir, exist_ok=True)
    got = 0
    for f in range(FOLDS):
        va_idx = np.where(fold == f)[0]
        cpath = os.path.join(ckdir, "fold%d.npz" % f)
        if os.path.isfile(cpath):
            d = np.load(cpath)
            vp, vr_d, tp, trg_d = d["vp"], d["vr"], d["tp"], d["trg"]
        elif agg_only:
            continue
        else:
            tr_idx = np.where(fold != f)[0]
            model, vp, vr = train_fold(tr_cache[tr_idx], tr_feats[tr_idx], tgt[tr_idx],
                                        tr_cache[va_idx], tr_feats[va_idx], fmean, fstd, pm, ps, device)
            vr_d = vr * reg_std + reg_mean
            tp, trg = infer_tta(model, te_cache, te_feats, fmean, fstd, pm, ps, device)
            trg_d = trg * reg_std + reg_mean
            np.savez(cpath, vp=vp, vr=vr_d, tp=tp, trg=trg_d)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        oof_probs[va_idx] = vp
        oof_regs[va_idx] = vr_d
        oof_mask[va_idx] = True
        test_probs += tp
        test_regs += trg_d
        got += 1

    if got == 0:
        raise RuntimeError("no folds available")
    test_probs /= got
    test_regs /= got

    m = oof_mask
    lam, oof_f1 = calibrate_class(oof_probs[m], oof_regs[m, GRAIN_IDX], labels[m], grain_mu, grain_sd)
    oof_pred_cls = apply_class(oof_probs[m], oof_regs[m, GRAIN_IDX], lam, grain_mu, grain_sd)
    test_pred_cls = apply_class(test_probs, test_regs[:, GRAIN_IDX], lam, grain_mu, grain_sd)

    final_test = np.zeros((len(te), 4))
    obs = []
    for j in range(4):
        cmean_oof = cmean[oof_pred_cls, j]
        w, a, b = calibrate_reg_col(oof_regs[m, j], cmean_oof, reg_raw[m, j])
        cmean_te = cmean[test_pred_cls, j]
        blended = w * test_regs[:, j] + (1 - w) * cmean_te
        final_test[:, j] = np.clip(a * blended + b, LOW[j], HIGH[j])
        rng = 760.0 if j == GRAIN_IDX else 2.0
        oof_bl = w * oof_regs[m, j] + (1 - w) * cmean[oof_pred_cls, j]
        oof_fin = np.clip(a * oof_bl + b, LOW[j], HIGH[j])
        obs.append(1 - np.sqrt(np.mean((oof_fin - reg_raw[m, j]) ** 2)) / rng)

    out = pd.DataFrame({"image": te["image"].values,
                        "prediction": test_pred_cls.astype(int),
                        REG_COLS[0]: final_test[:, 0],
                        REG_COLS[1]: final_test[:, 1],
                        REG_COLS[2]: final_test[:, 2],
                        REG_COLS[3]: final_test[:, 3]})
    os.makedirs("working", exist_ok=True)
    out.to_csv(os.path.join("working", "submission.csv"), index=False)
    raw = 0.6 * oof_f1 + 0.4 * float(np.mean(obs))
    print("done folds=%d oof_f1=%.4f lam=%.2f obs=%.4f raw=%.4f est_score=%.4f rows=%d" %
          (got, oof_f1, lam, float(np.mean(obs)), raw, 2 * raw - 1, len(out)))


if __name__ == "__main__":
    main()
