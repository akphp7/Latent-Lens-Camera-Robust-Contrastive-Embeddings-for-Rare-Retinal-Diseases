#!/usr/bin/env python
"""Latent Lens, step 4: trust layer + efficiency numbers. Needs latentlens_train.py in the same folder.

  !python latentlens_trust.py --stage feats    # unfamiliar-image score, calibration (ECE), retrieval; mean +- std over seeds
  !python latentlens_trust.py --stage umap     # UMAP figure coloured by camera and by disease
  !python latentlens_trust.py --stage stress   # blur / low-res / JPEG / dark / low-contrast / noise (needs GPU + the .pt files)
  !python latentlens_trust.py --stage eff      # parameters, FLOPs, CPU/GPU latency, training cost
  !python latentlens_trust.py --stage all      # everything, then writes results/trust_summary.md

Models compared everywhere: frozen DINOv2-S, plain SimCLR (--base, default b3) and Latent Lens (--final, default b3_fmix).
Every number comes from the saved features / checkpoints of real runs. Nothing is tuned on these metrics.
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import latentlens_train as T
from latentlens_train import say, OK, BAD, WARN, l2

R, FG = T.R, T.FG
NAMES = ['Frozen DINOv2-S', 'Plain SimCLR', 'Latent Lens']


def run_names(args, seed, hold=None):
    h = f'_hold-{hold}' if hold else ''
    return {'Frozen DINOv2-S': 'frozen', 'Plain SimCLR': f'{args.base}{h}_s{seed}', 'Latent Lens': f'{args.final}{h}_s{seed}'}


def load_feats(run):
    name = 'frozen_masked' if run == 'frozen' else run
    p = T.collect('features', '.npy').get(name)
    if p is None:
        sys.exit(f'{BAD} features for {name} not found. Attach the notebook outputs that trained it '
                 f'(and run `--variant {name.split("_hold")[0].rsplit("_s", 1)[0]}` first if it was never trained).')
    return np.load(p).astype(np.float32)


def load_meta():
    return pd.read_csv(os.path.join(T.find_cache(), 'meta.csv'))


def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg, pos]))


# ----------------------------------------------------------------------------- unfamiliar-image score
def unfamiliar(ref, q, k=5):
    """1 - mean cosine similarity to the k nearest reference images. Higher = more unfamiliar."""
    S = l2(q) @ l2(ref).T
    top = -np.partition(-S, k - 1, axis=1)[:, :k]
    return 1.0 - top.mean(1)


def ood_eval(F, meta, hold):
    rf = (meta.dataset.values == 'rfmid')
    cam, split = meta.camera.values, meta.split.values
    seen = rf & (cam != hold) if hold else rf
    ref, fam = F[seen & (split == 'train')], F[seen & (split == 'test')]
    s_fam = unfamiliar(ref, fam)
    out = {}
    idr = meta.dataset.values == 'idrid'
    if idr.any():
        out['idrid_vs_rfmid_test'] = auroc(s_fam, unfamiliar(ref, F[idr]))
    if hold:
        out['unseen_camera_vs_seen_test'] = auroc(s_fam, unfamiliar(ref, F[rf & (cam == hold)]))
    return out


# ----------------------------------------------------------------------------- calibration
def ece_binary(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def calib_eval(F, meta, hold):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize_scalar
    rf = meta[meta.dataset == 'rfmid'].reset_index(drop=True)
    Fr = l2(F[meta.dataset.values == 'rfmid'])
    cols = T.label_cols(rf)
    Y = rf[cols].fillna(0).values.astype(int)
    ok = (rf.camera.values != hold) if hold else np.ones(len(rf), bool)
    tr, va, te = [((rf.split == s).values & ok) for s in ('train', 'val', 'test')]
    sig = lambda z: 1 / (1 + np.exp(-z))
    before, after, temps = [], [], []
    for j, c in enumerate(cols):
        if Y[tr, j].sum() < 30 or Y[va, j].sum() < 5 or Y[te, j].sum() < 5:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Fr[tr], Y[tr, j])
        zv, zt = clf.decision_function(Fr[va]), clf.decision_function(Fr[te])

        def nll(t):
            p = np.clip(sig(zv / t), 1e-7, 1 - 1e-7)
            return -np.mean(Y[va, j] * np.log(p) + (1 - Y[va, j]) * np.log(1 - p))
        t_opt = float(minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x)
        before.append(ece_binary(sig(zt), Y[te, j]))
        after.append(ece_binary(sig(zt / t_opt), Y[te, j]))
        temps.append(t_opt)
    return dict(ece_before=float(np.mean(before)), ece_after=float(np.mean(after)),
                mean_temperature=float(np.mean(temps)), n_classes=len(before))


# ----------------------------------------------------------------------------- retrieval
def retrieval_eval(F, meta, hold, k=5):
    rf = meta[meta.dataset == 'rfmid'].reset_index(drop=True)
    Fr = l2(F[meta.dataset.values == 'rfmid'])
    cols = [c for c in T.label_cols(rf) if c != 'Disease_Risk']
    Y = rf[cols].fillna(0).values.astype(int)
    cam = np.asarray(rf.camera.values, dtype=object)
    ok = (cam != hold) if hold else np.ones(len(rf), bool)
    gal = (rf.split == 'train').values & ok
    qry = (cam == hold) if hold else (rf.split == 'test').values
    qry = qry & (Y.sum(1) > 0)
    Fq, Fg, Yq, Yg, cq, cg = Fr[qry], Fr[gal], Y[qry], Y[gal], cam[qry], cam[gal]
    match = (Yq @ Yg.T) > 0
    S = Fq @ Fg.T
    same = cq[:, None] == cg[None, :]

    def prec(Sm):
        top = np.argsort(-Sm, axis=1)[:, :k]
        return float(np.take_along_axis(match, top, 1).mean()), top
    out = {'n_queries': int(qry.sum()), 'chance': float(match.mean())}
    if hold:
        out['precision_at_k_unseen_camera'], _ = prec(S)
    else:
        out['precision_at_k_all'], top = prec(S)
        out['same_camera_share_top_k'] = float(np.take_along_axis(same, top, 1).mean())
        out['same_camera_share_base_rate'] = float(same.mean())
        out['precision_at_k_cross_camera'], _ = prec(np.where(same, -np.inf, S))
    return out


# ----------------------------------------------------------------------------- aggregate seeds
def msd(vals):
    v = np.array([x for x in vals if x is not None], float)
    return None if not len(v) else (float(v.mean()), float(v.std()))


def feats_stage(args):
    meta = load_meta()
    res = {}
    for name in NAMES:
        per = {}
        for hold in (None, args.hold):
            tag = 'hold' if hold else 'seen'
            runs = []
            for s in args.seeds:
                F = load_feats(run_names(args, s, hold)[name])
                runs.append({'ood': ood_eval(F, meta, hold), 'calib': calib_eval(F, meta, hold),
                             'retr': retrieval_eval(F, meta, hold)})
            per[tag] = runs
            say(OK, f'{name} [{tag}] done for seeds {args.seeds}')
        res[name] = per
    json.dump(res, open(os.path.join(R, 'trust_feats.json'), 'w'), indent=1)
    say(OK, f'wrote {R}/trust_feats.json')
    return res


# ----------------------------------------------------------------------------- UMAP figure
def umap_stage(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta = load_meta()
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(meta), min(1800, len(meta)), replace=False))
    sub = meta.iloc[idx]
    cams = sub.camera.values
    dis = sub['Disease_Risk'].values if 'Disease_Risk' in sub.columns else np.full(len(sub), np.nan)
    try:
        import umap
        reducer = lambda: umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0)
        method = 'UMAP'
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = lambda: PCA(2, random_state=0)
        method = 'PCA (umap-learn not installed)'
        say(WARN, 'umap-learn missing: using PCA. In Kaggle run: !pip -q install umap-learn')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.4), dpi=150)
    camcol = {'cam0': '#0072B2', 'cam1': '#E69F00', 'cam2': '#009E73', 'idrid': '#CC79A7'}
    rn = run_names(args, 0)
    for j, name in enumerate(NAMES):
        Z = reducer().fit_transform(l2(load_feats(rn[name])[idx]))
        for c, col in camcol.items():
            m = cams == c
            if m.any():
                ax[0, j].scatter(Z[m, 0], Z[m, 1], s=4, c=col, alpha=0.7, linewidths=0, rasterized=True,
                                 label=('IDRiD' if c == 'idrid' else c))
        lab = np.isnan(dis.astype(float))
        ax[1, j].scatter(Z[lab, 0], Z[lab, 1], s=4, c='#DDDDDD', alpha=0.6, linewidths=0, rasterized=True, label='IDRiD (no label)')
        for v, col, nm in ((0, '#0072B2', 'no disease flag'), (1, '#D55E00', 'disease risk')):
            m = (~lab) & (dis.astype(float) == v)
            ax[1, j].scatter(Z[m, 0], Z[m, 1], s=4, c=col, alpha=0.6, linewidths=0, rasterized=True, label=nm)
        ax[0, j].set_title(name, fontsize=10, fontweight='bold')
        for i in (0, 1):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for i, lbl in enumerate(('(a) coloured by camera', '(b) coloured by disease label')):
        ax[i, 0].set_ylabel(lbl, fontsize=9)
        h, l = ax[i, 2].get_legend_handles_labels()
        ax[i, 2].legend(h, l, markerscale=3, frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f'{method} of image embeddings, coloured by camera (top) and by disease label (bottom)', fontsize=11, y=0.995)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FG, f'umap_camera_vs_disease.{ext}'), dpi=300, bbox_inches='tight')
    say(OK, f'LOOK AT {FG}/umap_camera_vs_disease.png ({method})')


# ----------------------------------------------------------------------------- stress test (torch)
def degrade(X, kind):
    import cv2
    out = np.empty_like(X)
    rng = np.random.RandomState(0)
    for i, x in enumerate(X):
        if kind == 'blur_sigma2':
            y = cv2.GaussianBlur(x, (0, 0), 2)
        elif kind == 'blur_sigma4':
            y = cv2.GaussianBlur(x, (0, 0), 4)
        elif kind == 'low_res_4x':
            y = cv2.resize(cv2.resize(x, (56, 56), interpolation=cv2.INTER_AREA), (224, 224), interpolation=cv2.INTER_LINEAR)
        elif kind == 'jpeg_q20':
            _, b = cv2.imencode('.jpg', x[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 20])
            y = cv2.imdecode(b, 1)[:, :, ::-1]
        elif kind == 'dark_x0.4':
            y = np.clip(x.astype(np.float32) * 0.4, 0, 255)
        elif kind == 'low_contrast_x0.5':
            m = x.reshape(-1, 3).mean(0)
            y = np.clip(m + 0.5 * (x.astype(np.float32) - m), 0, 255)
        elif kind == 'noise_sd15':
            y = np.clip(x.astype(np.float32) + rng.normal(0, 15, x.shape), 0, 255)
        else:
            raise ValueError(kind)
        out[i] = np.asarray(y).astype(np.uint8)
    return out


KINDS = ['blur_sigma2', 'blur_sigma4', 'low_res_4x', 'jpeg_q20', 'dark_x0.4', 'low_contrast_x0.5', 'noise_sd15']


def stress_stage(args):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cdir = T.find_cache()
    X = np.load(os.path.join(cdir, 'images_224.npy'))
    meta = pd.read_csv(os.path.join(cdir, 'meta.csv'))
    rf = meta[meta.dataset == 'rfmid']
    cols = T.label_cols(rf)
    Y = rf[cols].fillna(0).values.astype(int)
    sp = rf.split.values
    tr, te = sp == 'train', sp == 'test'
    rf_pos = np.where(meta.dataset.values == 'rfmid')[0]
    te_pos = rf_pos[te]
    Xte = X[te_pos]
    mask = torch.from_numpy(T.circle_mask()).to(dev).view(1, 1, T.SIZE, T.SIZE)
    ckpts = T.collect('ckpt', '.pt')
    rn = run_names(args, 0)
    res = {}
    for name in NAMES:
        model = T.make_model(dev)
        mean, std = T.norm_stats(model, dev)
        if rn[name] != 'frozen':
            if rn[name] not in ckpts:
                sys.exit(f'{BAD} checkpoint {rn[name]}.pt not found. Attach the notebook output that trained it.')
            r = model.load_state_dict(torch.load(ckpts[rn[name]], map_location='cpu'), strict=False)
            if r.unexpected_keys:
                sys.exit(f'{BAD} unexpected keys in checkpoint: {r.unexpected_keys[:3]}')
        Ffile = load_feats(rn[name])
        Ftr = l2(Ffile[rf_pos[tr]])
        clean = T.extract(model, Xte, mean, std, mask, dev)
        diff = float(np.abs(l2(clean) - l2(Ffile[te_pos])).max())
        say(OK if diff < 0.05 else BAD, f'{name}: recomputed clean features match the saved ones (max abs diff {diff:.4f})')
        ref = Ftr
        rows = {}
        base = T.macro_auc(Ftr, Y[tr], l2(clean), Y[te], cols, 30, 5)['macro_auc']
        rows['clean'] = dict(macro_auc=base, unfamiliar_auroc=None)
        s_clean = unfamiliar(ref, clean)
        for kind in KINDS:
            Fd = T.extract(model, degrade(Xte, kind), mean, std, mask, dev)
            rows[kind] = dict(macro_auc=T.macro_auc(Ftr, Y[tr], l2(Fd), Y[te], cols, 30, 5)['macro_auc'],
                              unfamiliar_auroc=auroc(s_clean, unfamiliar(ref, Fd)))
            say(OK, f'{name:16s} {kind:18s} AUROC {rows[kind]["macro_auc"]:.3f} (clean {base:.3f}) | '
                    f'degraded-vs-clean unfamiliar AUROC {rows[kind]["unfamiliar_auroc"]:.3f}')
        res[name] = rows
        del model
    json.dump(res, open(os.path.join(R, 'trust_stress.json'), 'w'), indent=1)
    say(OK, f'wrote {R}/trust_stress.json')


# ----------------------------------------------------------------------------- efficiency (torch)
def eff_stage(args):
    import torch, timm
    from torch.utils.flop_counter import FlopCounterMode
    rows = {}
    makers = {'DINOv2-S backbone (Latent Lens, frozen DINOv2-S, SimCLR)': lambda: T.make_model('cpu'),
              'ResNet50 (reference only)': lambda: timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0)}
    for name, mk in makers.items():
        m = mk().eval()
        x = torch.randn(1, 3, T.SIZE, T.SIZE)
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                m(x)
            for _ in range(3):
                m(x)
            ts = []
            for _ in range(20):
                t = time.perf_counter(); m(x); ts.append(time.perf_counter() - t)
        r = dict(params_m=sum(p.numel() for p in m.parameters()) / 1e6, gflops=fc.get_total_flops() / 1e9,
                 cpu_ms_batch1=float(np.median(ts) * 1e3), cpu_threads=torch.get_num_threads())
        if torch.cuda.is_available():
            mg, xg = m.cuda(), x.cuda()
            with torch.no_grad():
                for _ in range(10):
                    mg(xg)
                torch.cuda.synchronize()
                ts = []
                for _ in range(50):
                    t = time.perf_counter(); mg(xg); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            r['gpu_ms_batch1'] = float(np.median(ts) * 1e3)
            r['gpu'] = torch.cuda.get_device_name(0)
        rows[name] = r
        say(OK, f'{name}: {r["params_m"]:.1f}M params, {r["gflops"]:.1f} GFLOPs, CPU {r["cpu_ms_batch1"]:.0f} ms/image')
    ck = T.collect('ckpt', '.pt').get(f'{args.final}_s0')
    if ck:
        sd = torch.load(ck, map_location='cpu')
        rows['trainable_params_m_latent_lens'] = sum(v.numel() for v in sd.values()) / 1e6
    secs = []
    for n, p in T.collect('results', '.json').items():
        if n.startswith(args.final + '_s') and '_hold' not in n:
            secs.append(json.load(open(p)).get('train_seconds'))
    secs = [s for s in secs if s]
    if secs:
        rows['train_seconds_median'] = float(np.median(secs))
        rows['train_runs'] = len(secs)
    json.dump(rows, open(os.path.join(R, 'trust_eff.json'), 'w'), indent=1)
    say(OK, f'wrote {R}/trust_eff.json')


# ----------------------------------------------------------------------------- summary
def fmt(v):
    return 'n/a' if v is None else f'{v[0]:.3f}' + (f'+-{v[1]:.3f}' if v[1] > 0 else '')


def summary_stage(args):
    L = ['# Latent Lens: trust layer and efficiency', '']
    p = os.path.join(R, 'trust_feats.json')
    if os.path.exists(p):
        d = json.load(open(p))
        g = lambda name, tag, sec, key: msd([r[sec].get(key) for r in d[name][tag]])
        L += ['## Unfamiliar-image score (kNN distance in embedding space), AUROC, higher = better', '',
              '| model | IDRiD vs RFMiD test (all cameras seen) | unseen camera cam1 vs seen test (cam1 held out) |', '|:--|--:|--:|']
        for n in NAMES:
            L.append(f'| {n} | {fmt(g(n,"seen","ood","idrid_vs_rfmid_test"))} | {fmt(g(n,"hold","ood","unseen_camera_vs_seen_test"))} |')
        L += ['', '## Calibration of per-class probes (ECE on test, lower = better; temperature fitted on validation)', '',
              '| model | ECE before | ECE after temperature scaling |', '|:--|--:|--:|']
        for n in NAMES:
            L.append(f'| {n} | {fmt(g(n,"seen","calib","ece_before"))} | {fmt(g(n,"seen","calib","ece_after"))} |')
        L += ['', '## Retrieval: do the 5 nearest training images share a disease with the query? (precision@5)', '',
              '| model | all gallery | cross-camera gallery only | same-camera share of top-5 (base rate ' +
              (f'{d[NAMES[0]]["seen"][0]["retr"]["same_camera_share_base_rate"]:.2f}' if d else 'n/a') + ') | unseen camera cam1 queries |',
              '|:--|--:|--:|--:|--:|']
        for n in NAMES:
            L.append(f'| {n} | {fmt(g(n,"seen","retr","precision_at_k_all"))} | {fmt(g(n,"seen","retr","precision_at_k_cross_camera"))} | '
                     f'{fmt(g(n,"seen","retr","same_camera_share_top_k"))} | {fmt(g(n,"hold","retr","precision_at_k_unseen_camera"))} |')
    p = os.path.join(R, 'trust_stress.json')
    if os.path.exists(p):
        d = json.load(open(p))
        L += ['', '## Robustness to image-quality shifts (RFMiD test, macro AUROC; probes trained on clean images)', '',
              '| condition | ' + ' | '.join(NAMES) + ' |', '|:--|' + '--:|' * len(NAMES)]
        for k in ['clean'] + KINDS:
            L.append(f'| {k} | ' + ' | '.join(f'{d[n][k]["macro_auc"]:.3f}' for n in NAMES) + ' |')
        L += ['', 'Degraded-vs-clean detection with the unfamiliar-image score (AUROC):', '',
              '| condition | ' + ' | '.join(NAMES) + ' |', '|:--|' + '--:|' * len(NAMES)]
        for k in KINDS:
            L.append(f'| {k} | ' + ' | '.join(f'{d[n][k]["unfamiliar_auroc"]:.3f}' for n in NAMES) + ' |')
    p = os.path.join(R, 'trust_eff.json')
    if os.path.exists(p):
        d = json.load(open(p))
        L += ['', '## Efficiency (224x224, one image)', '', '| model | params (M) | GFLOPs | CPU ms | GPU ms |', '|:--|--:|--:|--:|--:|']
        for k, v in d.items():
            if isinstance(v, dict):
                L.append(f'| {k} | {v["params_m"]:.1f} | {v["gflops"]:.1f} | {v["cpu_ms_batch1"]:.0f} ({v["cpu_threads"]} threads) | '
                         f'{v.get("gpu_ms_batch1", float("nan")):.1f} |')
        if 'trainable_params_m_latent_lens' in d:
            L.append(f'\nTrainable during fine-tuning: {d["trainable_params_m_latent_lens"]:.1f}M parameters.')
        if 'train_seconds_median' in d:
            L.append(f'Training time per run (median of {d["train_runs"]} runs): {d["train_seconds_median"]/60:.1f} min on the Kaggle GPU.')
    open(os.path.join(R, 'trust_summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    say(OK, f'wrote {R}/trust_summary.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['all', 'feats', 'umap', 'stress', 'eff', 'summary'])
    ap.add_argument('--base', default='b3')
    ap.add_argument('--final', default='b3_fmix')
    ap.add_argument('--hold', default='cam1')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    print(f'WORK={T.WORK}\n')
    if a.stage in ('all', 'feats'):
        feats_stage(a)
    if a.stage in ('all', 'umap'):
        umap_stage(a)
    if a.stage in ('all', 'stress'):
        stress_stage(a)
    if a.stage in ('all', 'eff'):
        eff_stage(a)
    if a.stage in ('all', 'summary'):
        summary_stage(a)
