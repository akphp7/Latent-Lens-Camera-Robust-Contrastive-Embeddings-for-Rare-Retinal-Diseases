#!/usr/bin/env python
"""Latent Lens, step 1: camera-aware contrastive fine-tuning + evaluation.

Needs the cache made by step 0 (images_224.npy + meta.csv). On Kaggle attach the output of your
"Latent Lens 0" notebook as an input; the script finds the cache by itself.

  !python latentlens_train.py --variant frozen              # B1: frozen DINOv2-S, same mask, no training
  !python latentlens_train.py --variant b3                  # B3: plain SimCLR, warm-started from DINOv2-S
  !python latentlens_train.py --variant b4                  # B4: Latent Lens (cross-camera positives,
                                                            #     false-negative mask, Fourier style swap)
  !python latentlens_train.py --variant b4 --hold cam1      # same, but cam1 is never seen in training
  !python latentlens_train.py --stage reeval                # re-score saved features after per-camera centering
  !python latentlens_train.py --stage report                # table of every finished run (mean +- std over seeds)

Variants: frozen, b3, b4, and ablations b4_no_nn (= b3 + fn + fmix), b4_no_fmix, b4_no_fn, plus the ladder b3_fn, b3_fmix.
Every run writes results/<run>.json. Smoke test: add --smoke (1 epoch, 3 steps).

Honest protocol, fixed before seeing results:
  * Contrastive training uses ONLY unlabeled RFMiD *train* images (and never the --hold camera).
  * Linear probes are fit on RFMiD train, scored on RFMiD val / test, on the held-out camera and on IDRiD.
  * Few-shot support sets are drawn with fixed seeds, so every variant sees the same support sets.
  * A circular retina mask is applied to every input of every model (also the frozen baseline).
"""
import os, re, sys, json, time, glob, random, argparse
import numpy as np
import pandas as pd

WORK = os.environ.get('LL_WORK', '/kaggle/working/latent-lens')
INPUT = os.environ.get('LL_INPUT', '/kaggle/input')
SIZE = 224
MODEL = 'vit_small_patch14_dinov2.lvd142m'
R, FE, CK, FG = [os.path.join(WORK, s) for s in ('results', 'features', 'ckpt', 'figs')]
for _p in (R, FE, CK, FG):
    os.makedirs(_p, exist_ok=True)
OK, BAD, WARN = '[ok]  ', '[FAIL]', '[warn]'

VARIANTS = {
    'frozen':     dict(train=False),
    'b3':         dict(train=True, nn=False, fn=False, fmix=False),
    'b4':         dict(train=True, nn=True,  fn=True,  fmix=True),
    'b4_no_nn':   dict(train=True, nn=False, fn=True,  fmix=True),
    'b4_no_fmix': dict(train=True, nn=True,  fn=True,  fmix=False),
    'b4_no_fn':   dict(train=True, nn=True,  fn=False, fmix=True),
    'b3_fn':      dict(train=True, nn=False, fn=True,  fmix=False),   # ladder: B3 + false-negative mask only
    'b3_fmix':    dict(train=True, nn=False, fn=False, fmix=True),    # ladder: B3 + Fourier style swap only
    # hybrid: Fourier style swap + supervised-contrastive term (images sharing a common-disease label are positives)
    'b3_fmix_sup': dict(train=True, nn=False, fn=False, fmix=True, sup=True),
}


def say(tag, msg):
    print(f'{tag} {msg}', flush=True)


# ----------------------------------------------------------------------------- data
def find_cache():
    local = os.path.join(WORK, 'data')
    if os.path.exists(os.path.join(local, 'images_224.npy')) and os.path.exists(os.path.join(local, 'meta.csv')):
        return local
    for root, _, names in os.walk(INPUT):
        if 'images_224.npy' in names and 'meta.csv' in names:
            return root
    sys.exit(f'{BAD} images_224.npy / meta.csv not found. In Kaggle: Add Input -> Notebook Output -> your '
             f'"Latent Lens 0" notebook (or run step 0 first).')


def circle_mask(size=SIZE, frac=0.98):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return (((yy - c) ** 2 + (xx - c) ** 2) <= (frac * size / 2) ** 2).astype(np.float32)


def l2(Fm):
    return Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-8)


def label_cols(rf):
    skip = {'path', 'stem', 'dataset', 'split', 'w', 'h', 'ID', 'size', 'camera', 'camera_source', 'dr_grade',
            'dme_grade', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'sharp', 'retina_w_frac'}
    cols = []
    for c in rf.columns:
        if c in skip:
            continue
        v = rf[c].dropna().unique()
        if len(v) and set(np.unique(v)).issubset({0, 1, 0.0, 1.0}):
            cols.append(c)
    return cols


# ----------------------------------------------------------------------------- probes (same as step 0)
def _pipe(C=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight='balanced'))


def camera_leakage(Fm, cams, seed=0):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    cams = np.asarray(cams)
    vc = pd.Series(cams).value_counts()
    if len(vc) < 2 or vc.min() < 5:
        return None
    pred = cross_val_predict(_pipe(), l2(Fm), cams, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return dict(acc=float(accuracy_score(cams, pred)), bal_acc=float(balanced_accuracy_score(cams, pred)),
                majority_rate=float(vc.max() / len(cams)), chance_bal_acc=float(1 / len(vc)), n_cameras=int(len(vc)))


def macro_auc(Ftr, Ytr, Fte, Yte, cols, min_pos_tr, min_pos_te):
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for j, c in enumerate(cols):
        if Ytr[:, j].sum() < min_pos_tr or Yte[:, j].sum() < min_pos_te or (1 - Yte[:, j]).sum() < min_pos_te:
            continue
        clf = _pipe().fit(Ftr, Ytr[:, j])
        aucs[c] = float(roc_auc_score(Yte[:, j], clf.predict_proba(Fte)[:, 1]))
    return dict(macro_auc=float(np.mean(list(aucs.values()))) if aucs else None, n_classes=len(aucs), per_class=aucs)


def few_shot(Fm, meta, cols, cam, shots=(1, 5, 10), n_seeds=10, min_pos=15):
    """K positives + K negatives per class from ONE camera, scored on the rest of that camera.
    Support sets use fixed seeds, so every variant is judged on identical support sets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rf = meta[(meta.dataset == 'rfmid') & (meta.camera == cam)]
    if len(rf) == 0:
        return {}
    Fc = l2(Fm[rf.index.values])
    Yc = rf[cols].fillna(0).values.astype(int)
    elig = [j for j in range(len(cols)) if Yc[:, j].sum() >= min_pos and (1 - Yc[:, j]).sum() >= min_pos]
    out = {}
    for K in shots:
        per_seed = []
        for s in range(n_seeds):
            rng = np.random.RandomState(1000 + s)
            aucs = []
            for j in elig:
                pos, neg = np.where(Yc[:, j] == 1)[0], np.where(Yc[:, j] == 0)[0]
                if len(pos) < K + 5 or len(neg) < K + 5:
                    continue
                sup = np.r_[rng.choice(pos, K, replace=False), rng.choice(neg, K, replace=False)]
                q = np.setdiff1d(np.arange(len(Fc)), sup)
                if Yc[q, j].sum() < 5 or (1 - Yc[q, j]).sum() < 5:
                    continue
                clf = LogisticRegression(C=10, max_iter=2000, class_weight='balanced')
                clf.fit(Fc[sup], np.r_[np.ones(K), np.zeros(K)])
                aucs.append(roc_auc_score(Yc[q, j], clf.decision_function(Fc[q])))
            if aucs:
                per_seed.append(float(np.mean(aucs)))
        if per_seed:
            out[str(K)] = dict(mean=float(np.mean(per_seed)), std=float(np.std(per_seed)),
                               n_classes=len(elig), camera=cam)
    return out


def center_by_camera(Fm, meta):
    """Subtract each camera's own mean embedding (cameras known from image size; no disease labels used).
    Applied identically to every model. For a held-out camera this uses its UNLABELED images (disclose it)."""
    Fn = l2(Fm).copy()
    cams = meta.camera.values
    for c in np.unique(cams):
        m = cams == c
        Fn[m] -= Fn[m].mean(0, keepdims=True)
    return Fn


def evaluate(Fm, meta, hold=None, center=False):
    from sklearn.metrics import roc_auc_score
    if center:
        Fm = center_by_camera(Fm, meta)
    res = {}
    rf_pos = np.where(meta.dataset.values == 'rfmid')[0]
    rf = meta.iloc[rf_pos].reset_index(drop=True)
    cols = label_cols(rf)
    Y = rf[cols].fillna(0).values.astype(int)
    Fr = l2(Fm[rf_pos])
    cams = rf.camera.values
    ok = (cams != hold) if hold else np.ones(len(rf), bool)
    tr = (rf.split == 'train').values & ok
    va = (rf.split == 'val').values & ok
    te = (rf.split == 'test').values & ok
    res['camera_leakage'] = camera_leakage(Fr, cams)
    res['val'] = macro_auc(Fr[tr], Y[tr], Fr[va], Y[va], cols, 30, 5)
    res['test'] = macro_auc(Fr[tr], Y[tr], Fr[te], Y[te], cols, 30, 5)
    if hold:
        h = cams == hold
        res['heldcam'] = macro_auc(Fr[tr], Y[tr], Fr[h], Y[h], cols, 20, 5)
        res['heldcam']['camera'] = hold
    # hidden-set transfer: RFMiD "DR" probe -> IDRiD (any DR = grade>=1, referable = grade>=2)
    idr = meta[meta.dataset == 'idrid']
    if len(idr) and 'DR' in cols and 'dr_grade' in idr.columns and idr.dr_grade.notna().any():
        j = cols.index('DR')
        clf = _pipe().fit(Fr[tr], Y[tr, j])
        s = clf.predict_proba(l2(Fm[idr.index.values]))[:, 1]
        g = idr.dr_grade.values
        res['idrid'] = dict(auc_dr_any=float(roc_auc_score(g >= 1, s)), auc_dr_referable=float(roc_auc_score(g >= 2, s)),
                            n=int(len(idr)), probe_trained_without=hold or 'nothing held out')
    res['few_shot'] = few_shot(Fm, meta, cols, hold or 'cam1')
    return res


# ----------------------------------------------------------------------------- neighbours
def build_nbrs(F0, P, cam_int, n_total, k=5):
    """Top-k most similar training images by FROZEN features: any camera, and a different camera only."""
    Fn = l2(F0[P])
    S = Fn @ Fn.T
    np.fill_diagonal(S, -2.0)
    na = np.argsort(-S, axis=1)[:, :k]
    S2 = S.copy()
    S2[cam_int[P][:, None] == cam_int[P][None, :]] = -2.0
    nc = np.argsort(-S2, axis=1)[:, :k]
    nbr_all = np.full((n_total, k), -1, np.int64)
    nbr_cross = np.full((n_total, k), -1, np.int64)
    nbr_all[P] = P[na]
    nbr_cross[P] = P[nc]
    return nbr_all, nbr_cross


# ----------------------------------------------------------------------------- torch parts
def make_model(dev):
    import timm
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE)
    return m.to(dev)


def norm_stats(model, dev):
    import torch, timm
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)
    return mean, std


def extract(model, X, mean, std, mask, dev, bs=128):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            xb = (xb * mask - mean) / std
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                f = model(xb)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def fourier_mix(x, xo, lam, beta=0.1):
    """Swap the low-frequency amplitude (global colour/illumination 'camera style') with another image's."""
    import torch
    fx, fo = torch.fft.fft2(x, dim=(-2, -1)), torch.fft.fft2(xo, dim=(-2, -1))
    ax = torch.fft.fftshift(fx.abs(), dim=(-2, -1))
    ao = torch.fft.fftshift(fo.abs(), dim=(-2, -1))
    ph = fx.angle()
    h = x.shape[-1]
    c, b = h // 2, max(int(beta * h), 1)
    sl = (slice(None), slice(None), slice(c - b, c + b + 1), slice(c - b, c + b + 1))
    ax[sl] = (1 - lam) * ax[sl] + lam * ao[sl]
    ax = torch.fft.ifftshift(ax, dim=(-2, -1))
    out = torch.fft.ifft2(ax * torch.exp(1j * ph)).real
    return out.clamp(0, 1)


def nce(anchor, cands, pos_col, ign, tau):
    import torch.nn.functional as F
    logits = (anchor @ cands.T) / tau
    logits = logits.masked_fill(ign, float('-inf'))
    return F.cross_entropy(logits, pos_col)


def compute_loss(z1, z2, z3, idx, pidx, nbr_all_t, cfg, tau):
    import torch
    B = z1.shape[0]
    ar = torch.arange(B, device=z1.device)

    def mask_for(src, extra=None):
        m = idx[:, None] == src[None, :]                       # same source image (duplicates in a batch)
        if cfg['fn']:                                          # frozen near-duplicates are not negatives
            nb = nbr_all_t[idx]
            m = m | (nb[:, :, None] == src[None, None, :]).any(1)
        if extra is not None:
            m = m | extra
        return m

    ign = mask_for(idx)
    ign[ar, ar] = False
    loss = 0.5 * (nce(z1, z2, ar, ign, tau) + nce(z2, z1, ar, ign, tau))
    if cfg['nn']:
        src = torch.cat([idx, pidx])
        ign2 = mask_for(src, extra=(pidx[:, None] == src[None, :]))
        ign2[ar, ar] = True                                    # the own second view is neither target nor negative
        ign2[ar, B + ar] = False                               # the cross-camera partner view is the target
        loss = loss + nce(z1, torch.cat([z2, z3]), B + ar, ign2, tau)
    return loss


def supcon_loss(z1, z2, y, idx, tau):
    """Supervised-contrastive loss. Positives of an image: its other view, any other copy of the same source image,
    and every image that shares at least one common-disease label. Images without labels only have their own other view."""
    import torch
    B = z1.shape[0]
    z = torch.cat([z1, z2])                                   # 2B x d, unit length
    yy = torch.cat([y, y])                                    # 2B x C (0/1)
    src = torch.cat([idx, idx])
    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    pos = (((yy @ yy.T) > 0) | (src[:, None] == src[None, :])) & ~eye
    logits = (z @ z.T / tau).masked_fill(eye, float('-inf'))
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return (-(logp.masked_fill(~pos, 0.0)).sum(1) / pos.sum(1)).mean()


def train(model, X, meta, P, cam_int, nbr_all, nbr_cross, cfg, args, dev, mean, std, mask, Yl=None):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import v2
    from torchvision.utils import save_image

    tf = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                     v2.RandomResizedCrop(SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
                     v2.RandomHorizontalFlip(),
                     v2.RandomRotation(20),
                     v2.ColorJitter(0.2, 0.2, 0.1, 0.02),
                     v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.0))], p=0.2)])

    class Views(Dataset):
        def __len__(self):
            return len(P)

        def __getitem__(self, k):
            i = int(P[k])
            p = int(random.choice(list(nbr_cross[i]))) if cfg['nn'] else i
            t = lambda j: torch.from_numpy(np.ascontiguousarray(X[j])).permute(2, 0, 1)
            return (tf(t(i)), tf(t(i)), (tf(t(p)) if cfg['nn'] else torch.zeros(1)), i, p,
                    (torch.from_numpy(np.ascontiguousarray(Yl[i])) if Yl is not None else torch.zeros(1)))

    # camera-balanced sampling (cam0 is 65% of RFMiD)
    cc = cam_int[P]
    w = 1.0 / pd.Series(cc).map(pd.Series(cc).value_counts()).values
    sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(P), replacement=True)
    nw = min(4, os.cpu_count() or 1)
    loader = DataLoader(Views(), batch_size=args.bs, sampler=sampler, num_workers=nw, drop_last=True,
                        persistent_workers=nw > 0)

    # trainable: last K blocks + final norm + projection head
    for p_ in model.parameters():
        p_.requires_grad = False
    train_params = []
    for blk in model.blocks[-args.k_blocks:]:
        train_params += list(blk.parameters())
    train_params += list(model.norm.parameters())
    for p_ in train_params:
        p_.requires_grad = True
    dim = model.num_features
    head = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 128)).to(dev)
    opt = torch.optim.AdamW([dict(params=train_params, lr=args.lr), dict(params=head.parameters(), lr=args.lr * 10)],
                            weight_decay=0.05)
    base = [g['lr'] for g in opt.param_groups]
    total = args.epochs * len(loader)
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(int(0.05 * total), 1)
    use_amp = dev == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    nbr_all_t = torch.from_numpy(nbr_all).to(dev)
    cam_t = torch.from_numpy(cam_int).to(dev)

    say(OK, f'training {args.variant}: {len(P)} images, batch {args.bs}, {len(loader)} steps/epoch, '
            f'{total} steps total, last {args.k_blocks} blocks trainable '
            f'({sum(p.numel() for p in train_params)/1e6:.1f}M params), nn={cfg["nn"]} fn={cfg["fn"]} fmix={cfg["fmix"]}')
    step, t0, losses, saved_fig = 0, time.time(), [], False
    model.train()
    while step < total:
        for a, b, c, idx, pidx, yl in loader:
            if step >= total:
                break
            lr_f = (step + 1) / warm if step < warm else 0.5 * (1 + np.cos(np.pi * (step - warm) / max(total - warm, 1)))
            for g, b0 in zip(opt.param_groups, base):
                g['lr'] = b0 * lr_f
            a, b = a.to(dev), b.to(dev)
            idx, pidx = idx.to(dev), pidx.to(dev)
            if cfg['nn']:
                c = c.to(dev)
                bad = (cam_t[idx] == cam_t[pidx]).sum().item()
                if bad:
                    sys.exit(f'{BAD} {bad} cross-camera partners share the anchor camera (bug)')
            if cfg['fmix']:
                cb = cam_t[idx]
                Rm = torch.rand(len(idx), len(idx), device=dev)
                Rm[cb[:, None] == cb[None, :]] = -1
                v, j = Rm.max(1)
                lam = (torch.rand(len(idx), 1, 1, 1, device=dev) * 0.7 + 0.3) * (v >= 0).float().view(-1, 1, 1, 1)
                a = fourier_mix(a, a[j], lam)
                Rm2 = torch.rand(len(idx), len(idx), device=dev)
                Rm2[cb[:, None] == cb[None, :]] = -1
                b = fourier_mix(b, b[Rm2.argmax(1)], lam)
            if not saved_fig:
                save_image(torch.cat([a[:4], b[:4], c[:4] if cfg['nn'] else a[:4]]).cpu(),
                           os.path.join(FG, f'aug_check_{args.variant}.png'), nrow=4)
                say(OK, f'LOOK AT {FG}/aug_check_{args.variant}.png: row1 view A, row2 view B (colour-style swapped), '
                        f'row3 cross-camera partner. Retinas must stay clearly visible.')
                saved_fig = True
            views = [a, b] + ([c] if cfg['nn'] else [])
            x = torch.cat(views)
            x = (x * mask - mean) / std
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                z = head(model(x))
            z = F.normalize(z.float(), dim=1)
            B = a.shape[0]
            z1, z2 = z[:B], z[B:2 * B]
            z3 = z[2 * B:] if cfg['nn'] else None
            if cfg.get('sup'):
                loss = supcon_loss(z1, z2, yl.to(dev), idx, args.tau)
            else:
                loss = compute_loss(z1, z2, z3, idx, pidx, nbr_all_t, cfg, args.tau)
            if not torch.isfinite(loss):
                sys.exit(f'{BAD} loss is {loss.item()} at step {step}. Lower --lr and retry.')
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            step += 1
            if step % 20 == 0 or step == 1 or step == total:
                say(OK, f'step {step}/{total} loss {np.mean(losses[-20:]):.3f} ({time.time()-t0:.0f}s)')
    model.eval()
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
               os.path.join(CK, f'{args.run}.pt'))
    n_tr = sum(p.numel() for p in train_params)
    return losses, n_tr


# ----------------------------------------------------------------------------- run / report
def run(args):
    import torch
    cfg = VARIANTS[args.variant]
    if args.smoke:
        args.epochs, args.max_steps = 1, 3
    args.run = f'{args.variant}' + (f'_hold-{args.hold}' if args.hold else '') + f'_s{args.seed}' + ('_smoke' if args.smoke else '')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    say(OK, f'run {args.run} on {dev}')
    if dev == 'cpu' and cfg['train'] and not args.smoke:
        say(WARN, 'no GPU: this will be very slow. Settings -> Accelerator -> GPU.')
    cdir = find_cache()
    X = np.load(os.path.join(cdir, 'images_224.npy'))
    meta = pd.read_csv(os.path.join(cdir, 'meta.csv'))
    assert len(X) == len(meta), f'cache mismatch: {len(X)} images vs {len(meta)} meta rows'
    say(OK, f'cache {cdir}: {X.shape}')
    cams_rf = sorted(c for c in meta.camera.unique() if str(c).startswith(('cam', 'cl')))
    cmap = {c: i for i, c in enumerate(cams_rf)}
    cam_int = meta.camera.map(cmap).fillna(-1).astype(int).values
    if args.hold and args.hold not in cmap:
        sys.exit(f'{BAD} --hold {args.hold} is not a camera. Cameras: {cams_rf}')

    P = np.where((meta.dataset.values == 'rfmid') & (meta.split.values == 'train') & (cam_int >= 0)
                 & (meta.camera.values != args.hold))[0]
    say(OK, 'pretraining images per camera: ' + ', '.join(f'{c}:{int((meta.camera.values[P]==c).sum())}' for c in cams_rf if c != args.hold))
    mask = torch.from_numpy(circle_mask()).to(dev).view(1, 1, SIZE, SIZE)

    model = make_model(dev)
    mean, std = norm_stats(model, dev)
    f0p = os.path.join(FE, 'frozen_masked.npy')
    if os.path.exists(f0p):
        F0 = np.load(f0p)
    else:
        t = time.time()
        F0 = extract(model, X, mean, std, mask, dev)
        np.save(f0p, F0)
        say(OK, f'frozen masked features {F0.shape} ({time.time()-t:.0f}s)')
    res = dict(run=args.run, variant=args.variant, hold=args.hold, seed=args.seed, cfg=cfg,
               params_total=int(sum(p.numel() for p in model.parameters())))
    if cfg['train']:
        use_nn = cfg['nn']
        if len(set(cam_int[P])) < 2 and use_nn:
            sys.exit(f'{BAD} need at least 2 cameras in the training set for cross-camera positives')
        nbr_all, nbr_cross = build_nbrs(F0, P, cam_int, len(meta))
        Yl = None
        if cfg.get('sup'):
            rfm = meta[meta.dataset == 'rfmid'].reset_index(drop=True)
            dcols = [c for c in label_cols(rfm) if c != 'Disease_Risk']
            Yfull = meta[dcols].fillna(0).values.astype(np.float32)
            keep = Yfull[P].sum(0) >= 30
            Yl = np.ascontiguousarray(Yfull[:, keep])
            say(OK, f'supervised-contrastive term uses {int(keep.sum())} common disease classes (train images of the held-out camera are never used): '
                    + ', '.join(c for c, k in zip(dcols, keep) if k))
            say(OK, f'{100 * float((Yl[P].sum(1) > 0).mean()):.0f}% of training images carry at least one of these labels')
        if use_nn:
            part = nbr_cross[P].ravel()
            same = (cam_int[np.repeat(P, nbr_cross.shape[1])] == cam_int[part]).mean()
            say(OK if same == 0 else BAD, f'cross-camera partners: {100*(1-same):.0f}% have a different camera than their anchor')
        t = time.time()
        losses, n_tr = train(model, X, meta, P, cam_int, nbr_all, nbr_cross, cfg, args, dev, mean, std, mask, Yl)
        res.update(train_seconds=time.time() - t, params_trainable=int(n_tr),
                   loss_first=float(np.mean(losses[:5])), loss_last=float(np.mean(losses[-5:])))
        say(OK if res['loss_last'] < res['loss_first'] else WARN, f'loss {res["loss_first"]:.3f} -> {res["loss_last"]:.3f}')
        F1 = extract(model, X, mean, std, mask, dev)
    else:
        F1 = F0
    if not np.isfinite(F1).all():
        sys.exit(f'{BAD} features contain NaN/inf')
    np.save(os.path.join(FE, f'{args.run}.npy'), F1.astype(np.float16))
    t = time.time()
    res.update(evaluate(F1, meta, args.hold))
    say(OK, f'evaluation done ({time.time()-t:.0f}s)')
    json.dump(res, open(os.path.join(R, f'{args.run}.json'), 'w'), indent=1)
    summary(res)


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def summary(r):
    f = lambda v: 'n/a' if v is None else f'{v:.3f}'
    say(OK, f'{r["run"]}: camera leakage {f(g(r,"camera_leakage","bal_acc"))} | val AUROC {f(g(r,"val","macro_auc"))} | '
            f'test AUROC {f(g(r,"test","macro_auc"))} | IDRiD DR {f(g(r,"idrid","auc_dr_any"))}' +
            (f' | held-out {r["hold"]} AUROC {f(g(r,"heldcam","macro_auc"))}' if r.get('hold') else ''))
    fs = r.get('few_shot') or {}
    if fs:
        say(OK, 'few-shot AUROC on ' + str(g(fs, '1', 'camera')) + ': ' +
            ', '.join(f'{k}-shot {fs[k]["mean"]:.3f}+-{fs[k]["std"]:.3f}' for k in fs))


RUN_RE = re.compile(r'^(?P<variant>[a-z0-9_]+?)(?:_hold-(?P<hold>cam\d+))?_s(?P<seed>\d+)$')


def collect(sub, ext):
    """Files in WORK/<sub> plus any <sub> folder under INPUT (outputs of earlier notebooks). WORK wins on name clashes."""
    found = {}
    dirs = [os.path.join(WORK, sub)]
    for root, ds, _ in os.walk(INPUT):
        if os.path.basename(root) == sub:
            dirs.append(root)
    for d in dirs[1:] + dirs[:1]:
        for p in glob.glob(os.path.join(d, '*' + ext)):
            found[os.path.basename(p)[:-len(ext)]] = p
    return found


def reeval():
    """Re-score every saved feature file after per-camera mean-centering. No training, no GPU needed."""
    cdir = find_cache()
    meta = pd.read_csv(os.path.join(cdir, 'meta.csv'))
    feats = collect('features', '.npy')
    done = collect('results', '.json')
    n = 0
    for name, p in sorted(feats.items()):
        m = RUN_RE.match(name)
        if not m or '_smoke' in name or name.endswith('_cc') or (name + '_cc') in done:
            continue
        t = time.time()
        F = np.load(p).astype(np.float32)
        if len(F) != len(meta):
            say(WARN, f'{name}: {len(F)} rows != {len(meta)}, skipped')
            continue
        hold = m.group('hold')
        res = dict(run=name + '_cc', variant=m.group('variant'), hold=hold, seed=int(m.group('seed')), centered=True)
        res.update(evaluate(F, meta, hold, center=True))
        json.dump(res, open(os.path.join(R, f'{name}_cc.json'), 'w'), indent=1)
        say(OK, f'{name} (camera-centered, {time.time()-t:.0f}s)')
        summary(res)
        n += 1
    if not n:
        say(WARN, 'nothing new to re-score (already done, or no feature files like b4_hold-cam1_s0.npy found)')


def report(hide_cc=False):
    rows = []
    for name, p in sorted(collect('results', '.json').items()):
        if name == 'poc0' or '_smoke' in name or name == 'summary':
            continue
        r = json.load(open(p))
        if 'variant' not in r or (hide_cc and r.get('centered')):
            continue
        fs = r.get('few_shot') or {}
        rows.append({'variant': r['variant'] + (' +camera-centering' if r.get('centered') else ''),
                     'held out': r.get('hold') or '-', 'seed': r.get('seed', 0),
                     'camera leak': g(r, 'camera_leakage', 'bal_acc'),
                     'val AUROC': g(r, 'val', 'macro_auc'), 'test AUROC': g(r, 'test', 'macro_auc'),
                     'held-cam AUROC': g(r, 'heldcam', 'macro_auc'), 'IDRiD DR AUROC': g(r, 'idrid', 'auc_dr_any'),
                     'IDRiD referable': g(r, 'idrid', 'auc_dr_referable'),
                     '1-shot': g(fs, '1', 'mean'), '5-shot': g(fs, '5', 'mean'), '10-shot': g(fs, '10', 'mean')})
    if not rows:
        sys.exit(f'{BAD} no finished runs in {R}')
    df = pd.DataFrame(rows).sort_values(['held out', 'variant', 'seed'])
    num = [c for c in df.columns if c not in ('variant', 'held out', 'seed')]
    agg = df.groupby(['held out', 'variant'])[num].agg(['mean', 'std'])
    cnt = df.groupby(['held out', 'variant']).size()
    out = []
    for (h, v), row in agg.iterrows():
        d = {'held out': h, 'variant': v, 'seeds': int(cnt[(h, v)])}
        for c in num:
            mu, sd = row[(c, 'mean')], row[(c, 'std')]
            d[c] = 'n/a' if pd.isna(mu) else (f'{mu:.3f}' if pd.isna(sd) else f'{mu:.3f}+-{sd:.3f}')
        out.append(d)
    t1 = pd.DataFrame(out).to_markdown(index=False)
    t2 = df.round(3).to_markdown(index=False)
    txt = ('# Latent Lens runs\n\n## Mean +- std over training seeds (std shown only when seeds > 1)\n\n' + t1 +
           '\n\n## Every run\n\n' + t2 + '\n')
    open(os.path.join(R, 'summary.md'), 'w').write(txt)
    print(t1)
    say(OK, f'wrote {R}/summary.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='run', choices=['run', 'report', 'reeval'])
    ap.add_argument('--variant', default='b4', choices=list(VARIANTS))
    ap.add_argument('--hold', default=None, help='camera never used for training or probe fitting, e.g. cam1')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--k_blocks', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--tau', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--hide_cc', action='store_true', help='report: leave out the camera-centering rows')
    a = ap.parse_args()
    print(f'WORK={WORK}\n')
    (report(a.hide_cc) if a.stage == 'report' else reeval()) if a.stage != 'run' else run(a)
