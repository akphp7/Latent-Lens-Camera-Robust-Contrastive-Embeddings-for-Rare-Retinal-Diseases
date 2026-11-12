#!/usr/bin/env python
"""Latent Lens, step 0: find the data, map cameras, cache images, run PoC-0.

Run on Kaggle (all defaults are already set):
    !python latentlens_poc0.py --stage all

Stages: audit -> cache -> frozen.  'all' runs the three in order.
Everything it writes goes to WORK (default /kaggle/working/latent-lens).
It looks for RFMiD and IDRiD anywhere under /kaggle/input, so folder names do not matter.
"""
import os, re, sys, json, time, argparse
import numpy as np
import pandas as pd
from PIL import Image

INPUT = os.environ.get('LL_INPUT', '/kaggle/input')
WORK = os.environ.get('LL_WORK', '/kaggle/working/latent-lens')
SIZE = 224
SEED = 0
EXPECT = {'train': 1920, 'val': 640, 'test': 640}   # official RFMiD split sizes
IMG_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
MIN_CAMERA_IMAGES = 20                               # smaller size groups are merged into 'other'

R = os.path.join(WORK, 'results')
F = os.path.join(WORK, 'features')
D = os.path.join(WORK, 'data')
G = os.path.join(WORK, 'figs')
for p in (R, F, D, G):
    os.makedirs(p, exist_ok=True)

OK, BAD, WARN = '[ok]  ', '[FAIL]', '[warn]'


def say(tag, msg):
    print(f'{tag} {msg}', flush=True)


# ----------------------------------------------------------------------------- audit
def rel(path):
    """Path below the Kaggle dataset folder, so the dataset's own name cannot confuse the matching."""
    parts = os.path.relpath(path, INPUT).replace('\\', '/').split('/')
    return '/'.join(parts[1:]) if len(parts) > 1 else parts[0]


def split_of(path):
    p = path.lower().replace('\\', '/')
    if 'eval' in p or 'valid' in p or re.search(r'(^|[^a-z])val([^a-z]|$)', p):
        return 'val'
    if 'test' in p:
        return 'test'
    if 'train' in p:
        return 'train'
    return 'unknown'


def dataset_of(path):
    p = path.lower().replace('\\', '/')
    if 'idrid' in p or 'disease grading' in p:
        return 'idrid'
    return 'rfmid'


def discover():
    imgs, csvs = [], []
    if not os.path.isdir(INPUT):
        sys.exit(f'{BAD} {INPUT} does not exist. On Kaggle, click Add Data and attach your RFMiD/IDRiD dataset.')
    for root, _, names in os.walk(INPUT):
        for n in names:
            full = os.path.join(root, n)
            low = n.lower()
            if low.endswith(IMG_EXT):
                imgs.append(full)
            elif low.endswith('.csv'):
                csvs.append(full)
    if not imgs:
        top = sorted(os.listdir(INPUT))[:20]
        sys.exit(f'{BAD} No images under {INPUT}. Top-level contents: {top}')
    return imgs, csvs


def read_labels(csvs):
    """RFMiD: csv with an ID column and 0/1 disease columns. IDRiD: csv with 'Image name'."""
    rf, idr = [], []
    for c in csvs:
        try:
            df = pd.read_csv(c, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(c, encoding='latin-1')
        df.columns = [str(x).strip() for x in df.columns]
        low = {x.lower(): x for x in df.columns}
        name = os.path.basename(c).lower()
        if 'image name' in low:
            d = pd.DataFrame({'stem': df[low['image name']].astype(str).str.replace(r'\.\w+$', '', regex=True)})
            for k, out in (('retinopathy grade', 'dr_grade'), ('risk of macular edema', 'dme_grade')):
                if k in low:
                    d[out] = df[low[k]].values
            d['split'] = split_of(rel(c))
            idr.append(d)
        elif 'id' in low and df.shape[1] >= 6:
            d = df.copy()
            d = d.rename(columns={low['id']: 'ID'})
            d['stem'] = d['ID'].astype(str).str.replace(r'\.\w+$', '', regex=True)
            d['split'] = split_of(rel(c))
            rf.append(d)
    rf = pd.concat(rf, ignore_index=True) if rf else pd.DataFrame()
    idr = pd.concat(idr, ignore_index=True) if idr else pd.DataFrame()
    return rf, idr


def audit():
    t0 = time.time()
    imgs, csvs = discover()
    say(OK, f'found {len(imgs)} images and {len(csvs)} csv files under {INPUT}')
    rows = []
    for p in imgs:
        r = rel(p)
        low = r.lower()
        ds = dataset_of(r)
        if ds == 'idrid' and ('segmentation' in low or 'localization' in low):
            continue                                     # same 516 photos appear in other IDRiD folders
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            say(WARN, f'cannot open {p}: {e}')
            continue
        rows.append(dict(path=p, stem=os.path.splitext(os.path.basename(p))[0], dataset=ds,
                         split=split_of(r), w=w, h=h))
    meta = pd.DataFrame(rows).drop_duplicates(subset=['dataset', 'stem', 'split']).reset_index(drop=True)

    rf_lab, idr_lab = read_labels(csvs)
    # attach labels
    rf = meta[meta.dataset == 'rfmid'].copy()
    idr = meta[meta.dataset == 'idrid'].copy()
    if len(rf_lab):
        rf_lab = rf_lab.drop_duplicates(subset=['split', 'stem'])
        rf = rf.merge(rf_lab.drop(columns=['ID'], errors='ignore'), on=['split', 'stem'], how='left')
    if len(idr_lab):
        idr_lab = idr_lab.drop_duplicates(subset=['split', 'stem'])
        idr = idr.merge(idr_lab, on=['split', 'stem'], how='left')
    meta = pd.concat([rf, idr], ignore_index=True)

    # camera from image size (RFMiD); IDRiD is a single Kowa camera
    meta['size'] = meta.w.astype(str) + 'x' + meta.h.astype(str)
    cam = pd.Series('', index=meta.index)
    rfm = meta.dataset == 'rfmid'
    counts = meta[rfm]['size'].value_counts()
    keep = list(counts[counts >= MIN_CAMERA_IMAGES].index)
    for i, s in enumerate(keep):
        cam[rfm & (meta['size'] == s)] = f'cam{i}'
    cam[rfm & (cam == '')] = 'other'
    cam[meta.dataset == 'idrid'] = 'idrid'
    meta['camera'] = cam
    camera_source = 'size' if len(keep) >= 2 else 'cluster'
    meta['camera_source'] = camera_source
    meta.to_csv(os.path.join(D, 'meta.csv'), index=False)

    # ---- report + self-checks
    L = ['# Data audit', '']
    L.append('## Images per dataset and split\n')
    L.append(pd.crosstab(meta.dataset, meta.split).to_markdown() + '\n')
    L.append('## Image sizes (RFMiD cameras are inferred from these)\n')
    L.append(pd.crosstab([meta.dataset, meta['size'], meta.camera], meta.split).to_markdown() + '\n')
    lab_cols = [c for c in rf.columns if c not in ('path', 'stem', 'dataset', 'split', 'w', 'h', 'ID')]
    if len(rf_lab):
        L.append('## RFMiD disease counts per camera (train split)\n')
        tr = meta[(meta.dataset == 'rfmid') & (meta.split == 'train')]
        num = [c for c in lab_cols if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
        if num:
            L.append(tr.groupby('camera')[num].sum().T.to_markdown() + '\n')

    n_rf = {s: int(((meta.dataset == 'rfmid') & (meta.split == s)).sum()) for s in ('train', 'val', 'test')}
    for s, n in EXPECT.items():
        say(OK if n_rf.get(s) == n else WARN, f'RFMiD {s}: {n_rf.get(s)} images (expected {n})')
    say(OK if len(rf_lab) else BAD, f'RFMiD label files found: {len(rf_lab)} rows')
    miss = int(meta[meta.dataset == 'rfmid'][lab_cols[0]].isna().sum()) if len(rf_lab) and lab_cols else -1
    if miss > 0:
        say(WARN, f'{miss} RFMiD images have no label row (check file names against the ID column)')
    n_idr = int((meta.dataset == 'idrid').sum())
    say(OK if n_idr else WARN, f'IDRiD: {n_idr} images (expected 516)')
    say(OK if camera_source == 'size' else WARN,
        f'camera groups from image size: {len(keep)} -> ' + ', '.join(f'{c}:{keep[i]}({counts[keep[i]]})' for i, c in enumerate([f"cam{j}" for j in range(len(keep))])))
    if camera_source == 'cluster':
        say(WARN, 'RFMiD images share one size, so cameras cannot be read from size. '
                  'The cache stage will fall back to clustering colour/noise statistics; treat the camera label as APPROXIMATE.')
    if n_idr and len(keep):
        idr_size = meta[meta.dataset == 'idrid']['size'].mode().iloc[0]
        same = [f'cam{i}' for i, s in enumerate(keep) if s == idr_size]
        say(OK, f'IDRiD size {idr_size} ' + (f'matches RFMiD {same[0]}, so IDRiD is NOT a new camera' if same else 'matches no RFMiD camera size'))
    open(os.path.join(R, 'audit.md'), 'w').write('\n'.join(L))
    say(OK, f'wrote {R}/audit.md and {D}/meta.csv  ({time.time()-t0:.0f}s)')
    return meta


# ----------------------------------------------------------------------------- cache
def _load_one(path):
    import cv2
    cv2.setNumThreads(1)
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    g = im.max(axis=2)
    small = g[::8, ::8] > 20
    ys, xs = np.where(small)
    if len(ys) < 100:
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        y0, y1 = max(ys.min() * 8, 0), min((ys.max() + 1) * 8, h)
        x0, x1 = max(xs.min() * 8, 0), min((xs.max() + 1) * 8, w)
    crop = im[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = max(ch, cw)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - ch) // 2:(s - ch) // 2 + ch, (s - cw) // 2:(s - cw) // 2 + cw] = crop
    out = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    m = g > 20
    px = im[m] if m.any() else im.reshape(-1, 3)
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    stats = [*px.mean(0), *px.std(0), cv2.Laplacian(gray, cv2.CV_64F).var(), (x1 - x0) / w]
    return out, stats


def cache(meta=None):
    from multiprocessing import Pool
    t0 = time.time()
    if meta is None:
        meta = pd.read_csv(os.path.join(D, 'meta.csv'))
    paths = meta.path.tolist()
    say(OK, f'caching {len(paths)} images at {SIZE}x{SIZE} (crop black border, pad to square) ...')
    with Pool(max(os.cpu_count() or 1, 1)) as pool:
        res = pool.map(_load_one, paths, chunksize=8)
    bad = [p for p, r in zip(paths, res) if r is None]
    if bad:
        sys.exit(f'{BAD} {len(bad)} images could not be read, first: {bad[0]}')
    X = np.stack([r[0] for r in res]).astype(np.uint8)
    st = np.array([r[1] for r in res], dtype=np.float32)
    for i, n in enumerate(['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'sharp', 'retina_w_frac']):
        meta[n] = st[:, i]
    if (meta.camera_source == 'cluster').any():
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        rfm = (meta.dataset == 'rfmid').values
        Z = StandardScaler().fit_transform(st[rfm][:, [0, 1, 2, 3, 4, 5, 6]])
        lab = KMeans(3, n_init=10, random_state=SEED).fit_predict(Z)
        meta.loc[rfm, 'camera'] = [f'cl{k}' for k in lab]
        say(WARN, 'camera labels now come from clustering (approximate)')
    np.save(os.path.join(D, f'images_{SIZE}.npy'), X)
    meta.to_csv(os.path.join(D, 'meta.csv'), index=False)
    # self-checks
    say(OK if X.mean() > 5 else BAD, f'mean pixel value {X.mean():.1f} (images are not blank)')
    say(OK, f'array {X.shape}, {X.nbytes/1e6:.0f} MB')
    # look-at-it montage: 5 images per camera
    import cv2
    cams = sorted(meta.camera.unique())
    rows = []
    rng = np.random.RandomState(SEED)
    for c in cams:
        idx = np.where(meta.camera.values == c)[0]
        pick = rng.choice(idx, min(5, len(idx)), replace=False)
        row = np.concatenate([X[i] for i in pick] + [np.zeros((SIZE, SIZE, 3), np.uint8)] * (5 - len(pick)), axis=1)
        cv2.putText(row, c, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(row)
    cv2.imwrite(os.path.join(G, 'samples_by_camera.png'), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    say(OK, f'LOOK AT {G}/samples_by_camera.png: five images per camera, retina should fill the frame ({time.time()-t0:.0f}s)')
    return meta, X


# ----------------------------------------------------------------------------- features
def feats_pixels(X):
    """Trivial colour-layout features. If the camera is readable from these, leakage is easy."""
    import cv2
    small = np.stack([cv2.resize(x, (8, 8), interpolation=cv2.INTER_AREA) for x in X]).reshape(len(X), -1) / 255.0
    return np.hstack([small, X.reshape(len(X), -1, 3).mean(1) / 255.0, X.reshape(len(X), -1, 3).std(1) / 255.0]).astype(np.float32)


TIMM_MODELS = {'resnet50_imagenet': 'resnet50.a1_in1k', 'dinov2_small': 'vit_small_patch14_dinov2.lvd142m'}


def feats_timm(name, X, bs=64):
    import torch, timm
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    kw = dict(pretrained=True, num_classes=0)
    if 'dinov2' in name:
        kw['img_size'] = SIZE
    m = timm.create_model(name, **kw).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            xb = (xb - mean) / std
            with torch.autocast(device_type=dev, enabled=(dev == 'cuda')):
                f = m(xb)
            out.append(f.float().cpu().numpy())
    n_par = sum(p.numel() for p in m.parameters())
    return np.concatenate(out), n_par


# ----------------------------------------------------------------------------- probes
def _pipe(C=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight='balanced'))


def l2(Fm):
    return Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-8)


def camera_leakage(Fm, cams):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    cams = np.asarray(cams)
    vc = pd.Series(cams).value_counts()
    if len(vc) < 2 or vc.min() < 5:
        return None
    pred = cross_val_predict(_pipe(), l2(Fm), cams, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return dict(acc=float(accuracy_score(cams, pred)), bal_acc=float(balanced_accuracy_score(cams, pred)),
                majority_rate=float(vc.max() / len(cams)), chance_bal_acc=float(1 / len(vc)), n_cameras=int(len(vc)))


def _label_cols(rf):
    skip = {'path', 'stem', 'dataset', 'split', 'w', 'h', 'ID', 'size', 'camera', 'camera_source', 'dr_grade', 'dme_grade',
            'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'sharp', 'retina_w_frac'}
    cols = []
    for c in rf.columns:
        if c in skip:
            continue
        v = rf[c].dropna().unique()
        if len(v) and set(np.unique(v)).issubset({0, 1, 0.0, 1.0}):
            cols.append(c)
    return cols


def _macro_auc(Ftr, Ytr, Fte, Yte, cols, min_pos_tr, min_pos_te):
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for j, c in enumerate(cols):
        if Ytr[:, j].sum() < min_pos_tr or Yte[:, j].sum() < min_pos_te or (1 - Yte[:, j]).sum() < min_pos_te:
            continue
        clf = _pipe().fit(Ftr, Ytr[:, j])
        aucs[c] = float(roc_auc_score(Yte[:, j], clf.predict_proba(Fte)[:, 1]))
    return aucs


def disease_and_loco(Fm, meta):
    rf = meta[meta.dataset == 'rfmid'].reset_index(drop=True)
    idx = np.where(meta.dataset.values == 'rfmid')[0]
    Fr = l2(Fm[idx])
    cols = _label_cols(rf)
    if not cols:
        return None, None
    Y = rf[cols].fillna(0).values.astype(int)
    tr, va = (rf.split == 'train').values, (rf.split == 'val').values
    out = {}
    a = _macro_auc(Fr[tr], Y[tr], Fr[va], Y[va], cols, 30, 5)      # fit on train, evaluate on VALIDATION only
    out = dict(val_macro_auc=float(np.mean(list(a.values()))) if a else None, n_classes=len(a), per_class=a)
    loco = {}
    for c in sorted(rf.camera.unique()):
        held = tr & (rf.camera == c).values
        rest = tr & (rf.camera != c).values
        if held.sum() < 50 or rest.sum() < 200:
            continue
        a = _macro_auc(Fr[rest], Y[rest], Fr[held], Y[held], cols, 20, 5)
        if a:
            loco[c] = dict(macro_auc=float(np.mean(list(a.values()))), n_classes=len(a), n_images=int(held.sum()))
    return out, loco


def frozen(meta=None, X=None):
    t0 = time.time()
    if meta is None:
        meta = pd.read_csv(os.path.join(D, 'meta.csv'))
    if X is None:
        X = np.load(os.path.join(D, f'images_{SIZE}.npy'))
    rfm = (meta.dataset == 'rfmid').values
    results = {}
    models = {'pixels_baseline': None}
    try:
        import torch, timm  # noqa
        models.update(TIMM_MODELS)
        say(OK, f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
        if not torch.cuda.is_available():
            say(WARN, 'no GPU: feature extraction will be slow. In Kaggle: Settings -> Accelerator -> GPU.')
    except ImportError:
        say(WARN, 'torch/timm not installed: only the pixel baseline will run. In Kaggle run: !pip -q install timm')
    for name, tname in models.items():
        try:
            fp = os.path.join(F, f'{name}.npy')
            n_par = None
            if os.path.exists(fp):
                Fm = np.load(fp)
            elif tname is None:
                Fm = feats_pixels(X)
            else:
                Fm, n_par = feats_timm(tname, X)
            np.save(fp, Fm)
        except Exception as e:
            say(BAD, f'{name}: {type(e).__name__}: {e}')
            if 'pretrained' in str(e).lower() or 'hub' in str(e).lower() or 'connection' in str(e).lower() or 'resolve' in str(e).lower():
                say(WARN, 'Looks like the weights could not be downloaded. In Kaggle turn Internet ON '
                          '(Settings -> Internet; needs a verified phone number) and rerun.')
            continue
        leak = camera_leakage(Fm[rfm], meta.camera.values[rfm])
        dis, loco = disease_and_loco(Fm, meta)
        results[name] = dict(dim=int(Fm.shape[1]), params=n_par, camera_leakage=leak, disease_val=dis, loco_train_only=loco)
        s = f'{name}: dim={Fm.shape[1]}'
        if leak:
            s += f' | camera guess bal.acc={leak["bal_acc"]:.3f} (chance {leak["chance_bal_acc"]:.2f}, majority {leak["majority_rate"]:.2f})'
        if dis and dis['val_macro_auc'] is not None:
            s += f' | disease val macro-AUROC={dis["val_macro_auc"]:.3f} over {dis["n_classes"]} classes'
        say(OK, s)
    json.dump(results, open(os.path.join(R, 'poc0.json'), 'w'), indent=2)

    # Gate G0 (proposed thresholds, written down before seeing results)
    real = {k: r for k, r in results.items() if k != 'pixels_baseline' and r.get('camera_leakage')}
    pool = real if real else {k: r for k, r in results.items() if r.get('camera_leakage')}
    if not real:
        say(WARN, 'only the trivial pixel baseline ran, so G0 below is NOT yet meaningful; rerun with torch and timm installed')
    best = max((r['camera_leakage']['bal_acc'] for r in pool.values()), default=None)
    print('\n==== GATE G0 ====')
    if best is None:
        say(BAD, 'camera probe could not run (need at least 2 camera groups with 5+ images)')
    elif best >= 0.9:
        say(OK, f'G0 PASS: camera is easy to read from standard features (best balanced accuracy {best:.2f}). The problem is real in this data.')
    elif best >= 0.7:
        say(WARN, f'G0 PARTIAL: camera is moderately readable ({best:.2f}). Continue, but frame the claim carefully.')
    else:
        say(BAD, f'G0 WEAK: camera is hard to read ({best:.2f}). Tell Claude before training anything; the story needs to change.')
    say(OK, f'wrote {R}/poc0.json ({time.time()-t0:.0f}s)')
    return results


# ----------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['audit', 'cache', 'frozen', 'all'])
    a = ap.parse_args()
    np.random.seed(SEED)
    print(f'INPUT={INPUT}\nWORK={WORK}\n')
    if a.stage in ('audit', 'all'):
        meta = audit()
    if a.stage in ('cache', 'all'):
        meta, X = cache(meta if a.stage == 'all' else None)
    if a.stage in ('frozen', 'all'):
        frozen(meta if a.stage == 'all' else None, X if a.stage == 'all' else None)
