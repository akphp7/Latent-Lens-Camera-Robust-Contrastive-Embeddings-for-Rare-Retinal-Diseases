#!/usr/bin/env python
"""Latent Lens, step 5: rare-disease evidence, better unfamiliar score, supervised baseline, camera hold-out summary.
Needs latentlens_train.py and latentlens_trust.py in the same folder.

  !python latentlens_step5.py --stage sup --seed 0         # supervised baseline (labels of the common classes only), GPU
  !python latentlens_step5.py --stage rare                 # rare vs common diseases, few-shot on rare ones, paired confidence intervals
  !python latentlens_step5.py --stage ood2                 # unfamiliar score: kNN distance vs Mahalanobis distance
  !python latentlens_step5.py --stage loco                 # table for every held-out camera (cam0, cam1, cam2)

Definitions, fixed before looking at any result:
  common class = at least 30 positive training images (this is what every earlier table used);
  rare class   = 8 to 29 positive training images AND at least 4 positives in val+test together.
  Rare classes are scored on val+test pooled (too few positives in test alone). Probes are fit on train only.
Every number comes from saved features of real runs.
"""
import os, sys, json, time, random, argparse
import numpy as np
import pandas as pd
import latentlens_train as T
import latentlens_trust as U
from latentlens_train import say, OK, BAD, WARN, l2

R, FE, CK = T.R, T.FE, T.CK
MODELS = [('Frozen DINOv2-S', 'frozen'), ('Plain SimCLR', 'b3'), ('Latent Lens', 'b3_fmix'), ('Supervised fine-tune', 'sup'),
          ('Latent Lens + labels', 'b3_fmix_sup')]
SHOTS = (3, 5, 10)


def fast_auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def setup(args):
    meta = U.load_meta()
    rf_mask = meta.dataset.values == 'rfmid'
    rf = meta[rf_mask].reset_index(drop=True)
    cols = T.label_cols(rf)
    Y = rf[cols].fillna(0).values.astype(int)
    have = T.collect('features', '.npy')
    feats = {}
    for name, var in MODELS:
        seeds = [0] if var == 'frozen' else args.seeds
        key = lambda s: 'frozen_masked' if var == 'frozen' else f'{var}_s{s}'
        if not all(key(s) in have for s in seeds):
            if var in ('sup', 'b3_fmix_sup'):
                say(WARN, f'no features found for {name}: that row is left out')
                continue
            sys.exit(f'{BAD} features missing for {name}: need {[key(s) for s in seeds]}')
        feats[name] = [l2(np.load(have[key(s)]).astype(np.float32)[rf_mask]) for s in seeds]
    return meta, rf, cols, Y, feats


def probe_scores(Ftr, ytr, Fev):
    clf = T._pipe().fit(Ftr, ytr)
    return clf.predict_proba(Fev)[:, 1]


def msd(v):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    return (float(v.mean()), float(v.std())) if len(v) else (np.nan, 0.0)


def fmt(m, s=None):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return 'n/a'
    return f'{m:.3f}' + (f'+-{s:.3f}' if s else '')


# ----------------------------------------------------------------------------- rare vs common
def rare_stage(args):
    meta, rf, cols, Y, feats = setup(args)
    sp = rf.split.values
    tr, te, pool = sp == 'train', sp == 'test', sp != 'train'
    ntr = Y[tr].sum(0)
    common = [j for j in range(len(cols)) if ntr[j] >= 30 and Y[te, j].sum() >= 5 and (1 - Y[te, j]).sum() >= 5]
    rare = [j for j in range(len(cols)) if 8 <= ntr[j] < 30 and Y[pool, j].sum() >= 4]
    dropped = [(cols[j], int(ntr[j])) for j in range(len(cols)) if j not in common and j not in rare and ntr[j] > 0]
    say(OK, f'{len(common)} common classes, {len(rare)} rare classes: ' + ', '.join(f'{cols[j]}({ntr[j]})' for j in rare))
    say(OK, f'{len(dropped)} classes too small to evaluate (left out, disclosed): ' + ', '.join(f'{c}({n})' for c, n in dropped))
    names = list(feats)
    rng = np.random.RandomState(0)

    # scores: sc[name][seed_idx][class_pos] -> scores over the eval set
    def all_scores(classes, ev):
        sc = {}
        for n in names:
            sc[n] = []
            for Fm in feats[n]:
                sc[n].append([probe_scores(Fm[tr], Y[tr, j], Fm[ev]) for j in classes])
        return sc
    t = time.time()
    sc_rare = all_scores(rare, pool)
    sc_com = all_scores(common, te)
    say(OK, f'probes fitted ({time.time()-t:.0f}s)')

    def table(sc, classes, ev):
        yv = np.stack([Y[ev, j] for j in classes])
        per = {n: np.array([[fast_auc(yv[k], sc[n][s][k]) for k in range(len(classes))] for s in range(len(sc[n]))]) for n in names}
        return yv, per
    yv_r, per_r = table(sc_rare, rare, pool)
    yv_c, per_c = table(sc_com, common, te)

    # paired bootstrap over evaluation images (same resample for every model)
    def boot(sc, yv, B):
        N = yv.shape[1]
        out = {n: [] for n in names}
        for _ in range(B):
            idx = rng.randint(0, N, N)
            ys = yv[:, idx]
            valid = [k for k in range(len(ys)) if 0 < ys[k].sum() < N]
            if not valid:
                continue
            for n in names:
                v = [np.nanmean([fast_auc(ys[k], sc[n][s][k][idx]) for k in valid]) for s in range(len(sc[n]))]
                out[n].append(np.mean(v))
        return {n: np.array(v) for n, v in out.items()}
    t = time.time()
    b_r, b_c = boot(sc_rare, yv_r, args.boot), boot(sc_com, yv_c, args.boot)
    say(OK, f'bootstrap done ({time.time()-t:.0f}s)')

    def ci(a, b):
        d = b_arr[b] - b_arr[a]
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())

    L = ['# Latent Lens step 5: rare vs common diseases', '',
         f'Common classes (>=30 train positives, scored on test): {len(common)}. Rare classes (8-29 train positives, scored on val+test pooled): {len(rare)}.',
         'Classes with fewer than 8 training positives cannot be evaluated and are left out: ' + ', '.join(f'{c} ({n})' for c, n in dropped) + '.', '']
    res = {'common': [cols[j] for j in common], 'rare': [cols[j] for j in rare], 'left_out': dropped}
    for title, per, bt in (('RARE classes', per_r, b_r), ('COMMON classes', per_c, b_c)):
        L += [f'## {title}: macro AUROC, linear probe fit on all training images (mean +- std over training seeds)', '',
              '| model | macro AUROC | 95% bootstrap interval of the mean over resampled eval images |', '|:--|--:|--:|']
        for n in names:
            m = per[n].mean(1)
            lo, hi = np.percentile(bt[n], [2.5, 97.5])
            L.append(f'| {n} | {fmt(*msd(m))} | {lo:.3f} to {hi:.3f} |')
            res.setdefault(title, {})[n] = dict(mean=msd(m)[0], std=msd(m)[1], ci=[float(lo), float(hi)])
        b_arr = bt
        L += ['', 'Paired differences (same resampled images for both models; "P(better)" = share of resamples where the first model wins):', '',
              '| comparison | mean difference | 95% interval | P(better) |', '|:--|--:|--:|--:|']
        for a, b in (('Frozen DINOv2-S', 'Latent Lens'), ('Plain SimCLR', 'Latent Lens'), ('Frozen DINOv2-S', 'Plain SimCLR'),
                     ('Supervised fine-tune', 'Latent Lens'), ('Latent Lens', 'Latent Lens + labels'),
                     ('Supervised fine-tune', 'Latent Lens + labels'), ('Frozen DINOv2-S', 'Latent Lens + labels')):
            if a in names and b in names:
                d, lo, hi, p = ci(a, b)
                L.append(f'| {b} minus {a} | {d:+.3f} | {lo:+.3f} to {hi:+.3f} | {p:.2f} |')
                res.setdefault(title + '_paired', {})[f'{b} - {a}'] = dict(diff=d, lo=lo, hi=hi, p_better=p)
        L.append('')
    L += ['## Per rare class (AUROC, mean over training seeds)', '',
          '| class | train positives | val+test positives | ' + ' | '.join(names) + ' |', '|:--|--:|--:|' + '--:|' * len(names)]
    for k, j in enumerate(rare):
        L.append(f'| {cols[j]} | {int(ntr[j])} | {int(Y[pool, j].sum())} | ' + ' | '.join(f'{per_r[n][:, k].mean():.3f}' for n in names) + ' |')

    # few-shot on the rare classes: K positives + K negatives from the TRAIN images, scored on val+test
    from sklearn.linear_model import LogisticRegression
    L += ['', '## Few-shot on rare classes (K positive + K negative training images, scored on val+test; 10 fixed support sets, identical for every model)', '',
          '| K | ' + ' | '.join(names) + ' | classes |', '|--:|' + '--:|' * len(names) + '--:|']
    fs_res = {}
    for K in SHOTS:
        classes = [j for j in rare if ntr[j] >= K]
        grid = {n: np.full((len(feats[n]), args.support_sets, len(classes)), np.nan) for n in names}
        for s in range(args.support_sets):
            for kk, j in enumerate(classes):
                r = np.random.RandomState(1000 + s * 100 + j)
                pos, neg = np.where(tr & (Y[:, j] == 1))[0], np.where(tr & (Y[:, j] == 0))[0]
                sup = np.r_[r.choice(pos, K, replace=False), r.choice(neg, K, replace=False)]
                for n in names:
                    for si, Fm in enumerate(feats[n]):
                        clf = LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(Fm[sup], np.r_[np.ones(K), np.zeros(K)])
                        grid[n][si, s, kk] = fast_auc(Y[pool, j], clf.decision_function(Fm[pool]))
        row = []
        for n in names:
            per_set = np.nanmean(grid[n], axis=(0, 2))              # one number per support set (mean over seeds, classes)
            row.append(fmt(per_set.mean(), per_set.std()))
        L.append(f'| {K} | ' + ' | '.join(row) + f' | {len(classes)} |')
        fs_res[str(K)] = {n: float(np.nanmean(grid[n])) for n in names}
        if 'Frozen DINOv2-S' in names and 'Latent Lens' in names:
            d = np.nanmean(grid['Latent Lens'], axis=0) - np.nanmean(grid['Frozen DINOv2-S'], axis=0)   # support set x class
            cl = np.nanmean(d, axis=0)
            bs = [np.mean(cl[rng.randint(0, len(cl), len(cl))]) for _ in range(2000)]
            fs_res[str(K)]['latent_minus_frozen'] = dict(mean=float(np.nanmean(d)), win_rate=float((d > 0).mean()),
                                                        ci_over_classes=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    L += ['', 'Latent Lens minus frozen, per (class, support set) pair:', '', '| K | mean difference | share of pairs where Latent Lens is better | 95% interval (resampling classes) |', '|--:|--:|--:|--:|']
    for K in SHOTS:
        d = fs_res[str(K)].get('latent_minus_frozen')
        if d:
            L.append(f'| {K} | {d["mean"]:+.3f} | {d["win_rate"]:.2f} | {d["ci_over_classes"][0]:+.3f} to {d["ci_over_classes"][1]:+.3f} |')
    res['few_shot_rare'] = fs_res

    # the nearly empty image seen in the augmentation check
    sd = [c for c in ('std_r', 'std_g', 'std_b') if c in meta.columns]
    if sd:
        m = meta.assign(std_mean=meta[sd].mean(1)).sort_values('std_mean')
        L += ['', '## Lowest-contrast images in the data (check for near-empty pictures)', '', '| row | dataset | split | camera | mean pixel std |', '|--:|:--|:--|:--|--:|']
        for i, r in m.head(8).iterrows():
            L.append(f'| {i} | {r["dataset"]} | {r["split"]} | {r["camera"]} | {r["std_mean"]:.1f} |')
    open(os.path.join(R, 'rare_summary.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(res, open(os.path.join(R, 'rare.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))
    say(OK, f'wrote {R}/rare_summary.md and rare.json')


# ----------------------------------------------------------------------------- unfamiliar score, second method
def ood2_stage(args):
    from sklearn.covariance import LedoitWolf
    meta = U.load_meta()
    rfm = meta.dataset.values == 'rfmid'
    cam, split = meta.camera.values, meta.split.values
    idr = meta.dataset.values == 'idrid'

    def maha(ref, q):
        lw = LedoitWolf().fit(ref)
        d = q - lw.location_
        return np.einsum('ij,jk,ik->i', d, lw.precision_, d)

    def one(F, hold):
        F = l2(F)
        seen = rfm & (cam != hold) if hold else rfm
        ref, fam = F[seen & (split == 'train')], F[seen & (split == 'test')]
        out = {}
        for nm, fn in (('knn', lambda r, q: U.unfamiliar(r, q)), ('mahalanobis', maha)):
            s0 = fn(ref, fam)
            if idr.any():
                out[f'{nm}_idrid'] = U.auroc(s0, fn(ref, F[idr]))
            if hold:
                out[f'{nm}_unseen_camera'] = U.auroc(s0, fn(ref, F[rfm & (cam == hold)]))
        return out
    res, rows = {}, []
    for name, var in MODELS[:3]:
        for hold in (None, args.hold):
            seeds = [0] if var == 'frozen' else args.seeds
            vals = []
            for s in seeds:
                run = 'frozen' if var == 'frozen' else f'{var}{"_hold-" + hold if hold else ""}_s{s}'
                vals.append(one(U.load_feats(run), hold))
            res[f'{name} [{"hold " + hold if hold else "all seen"}]'] = vals
            say(OK, f'{name} [{hold or "seen"}] done')
    L = ['# Unfamiliar-image score: kNN distance vs Mahalanobis distance (AUROC, higher = better; 0.5 = chance)', '',
         '| model | kNN: IDRiD vs RFMiD test | Mahalanobis: IDRiD vs RFMiD test | kNN: unseen camera cam1 | Mahalanobis: unseen camera cam1 |', '|:--|--:|--:|--:|--:|']
    for name, _ in MODELS[:3]:
        a, b = res[f'{name} [all seen]'], res[f'{name} [hold {args.hold}]']
        g = lambda lst, k: fmt(*msd([x.get(k) for x in lst]))
        L.append(f'| {name} | {g(a, "knn_idrid")} | {g(a, "mahalanobis_idrid")} | {g(b, "knn_unseen_camera")} | {g(b, "mahalanobis_unseen_camera")} |')
    open(os.path.join(R, 'ood2_summary.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(res, open(os.path.join(R, 'ood2.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))
    say(OK, f'wrote {R}/ood2_summary.md')


# ----------------------------------------------------------------------------- every held-out camera
def loco_stage(args):
    done = T.collect('results', '.json')
    L = ['# Held-out camera: frozen vs plain SimCLR vs Latent Lens (macro AUROC on the camera that was never used for training or probe fitting)', '',
         '| held-out camera | model | seeds | held-camera AUROC | 5-shot on that camera | IDRiD DR AUROC | RFMiD test AUROC |', '|:--|:--|--:|--:|--:|--:|--:|']
    per = {}
    for hold in ('cam0', 'cam1', 'cam2'):
        for name, var in MODELS:
            runs = [json.load(open(p)) for n, p in done.items() if n.startswith(f'{var}_hold-{hold}_s') and not n.endswith('_cc')]   # _cc = camera-centering diagnostic, not a separate seed
            if not runs:
                continue
            g = T.g
            h = [g(r, 'heldcam', 'macro_auc') for r in runs]
            f5 = [g(r, 'few_shot', '5', 'mean') for r in runs]
            idr = [g(r, 'idrid', 'auc_dr_any') for r in runs]
            te = [g(r, 'test', 'macro_auc') for r in runs]
            per[(hold, name)] = h
            L.append(f'| {hold} | {name} | {len(runs)} | {fmt(*msd(h))} | {fmt(*msd(f5))} | {fmt(*msd(idr))} | {fmt(*msd(te))} |')
    L += ['', '## Latent Lens versus the others, per seed (held-camera AUROC difference)', '', '| held-out camera | comparison | per-seed differences | seeds where Latent Lens is better |', '|:--|:--|:--|--:|']
    for hold in ('cam0', 'cam1', 'cam2'):
        for other in ('Frozen DINOv2-S', 'Plain SimCLR'):
            a, b = per.get((hold, 'Latent Lens')), per.get((hold, other))
            if a and b:
                if len(b) == 1:
                    b = b * len(a)
                d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
                L.append(f'| {hold} | Latent Lens minus {other} | ' + ', '.join(f'{x:+.3f}' for x in d) + f' | {sum(x > 0 for x in d)}/{len(d)} |')
    open(os.path.join(R, 'loco_summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    say(OK, f'wrote {R}/loco_summary.md')


# ----------------------------------------------------------------------------- supervised baseline (torch)
def sup_stage(args):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import v2
    if args.smoke:
        args.epochs, args.max_steps = 1, 3
    run = 'sup' + (f'_hold-{args.hold}' if args.hold else '') + f'_s{args.seed}' + ('_smoke' if args.smoke else '')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    say(OK, f'run {run} on {dev}')
    cdir = T.find_cache()
    X = np.load(os.path.join(cdir, 'images_224.npy'))
    meta = pd.read_csv(os.path.join(cdir, 'meta.csv'))
    rf_all = meta[meta.dataset == 'rfmid']
    cols = T.label_cols(rf_all.reset_index(drop=True))
    cams = sorted(c for c in meta.camera.unique() if str(c).startswith(('cam', 'cl')))
    cmap = {c: i for i, c in enumerate(cams)}
    cam_int = meta.camera.map(cmap).fillna(-1).astype(int).values
    P = np.where((meta.dataset.values == 'rfmid') & (meta.split.values == 'train') & (cam_int >= 0) & (meta.camera.values != args.hold))[0]
    Yall = meta[cols].fillna(0).values.astype(np.float32)
    sup_cols = [j for j in range(len(cols)) if Yall[P, j].sum() >= 30]
    say(OK, f'supervised on {len(P)} train images, {len(sup_cols)} common classes (rare classes are NOT supervised)')
    Ys = Yall[:, sup_cols]
    pw = torch.from_numpy(np.clip((len(P) - Ys[P].sum(0)) / np.maximum(Ys[P].sum(0), 1), 1, 10)).float().to(dev)
    mask = torch.from_numpy(T.circle_mask()).to(dev).view(1, 1, T.SIZE, T.SIZE)
    model = T.make_model(dev)
    mean, std = T.norm_stats(model, dev)
    tf = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.RandomResizedCrop(T.SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
                     v2.RandomHorizontalFlip(), v2.RandomRotation(20), v2.ColorJitter(0.2, 0.2, 0.1, 0.02)])

    class DS(Dataset):
        def __len__(self):
            return len(P)

        def __getitem__(self, k):
            i = int(P[k])
            return tf(torch.from_numpy(np.ascontiguousarray(X[i])).permute(2, 0, 1)), torch.from_numpy(Ys[i])
    cc = cam_int[P]
    w = 1.0 / pd.Series(cc).map(pd.Series(cc).value_counts()).values
    nw = min(4, os.cpu_count() or 1)
    loader = DataLoader(DS(), batch_size=args.bs, sampler=WeightedRandomSampler(torch.from_numpy(w).double(), len(P), replacement=True),
                        num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    for p_ in model.parameters():
        p_.requires_grad = False
    tp = []
    for blk in model.blocks[-args.k_blocks:]:
        tp += list(blk.parameters())
    tp += list(model.norm.parameters())
    for p_ in tp:
        p_.requires_grad = True
    head = nn.Linear(model.num_features, len(sup_cols)).to(dev)
    opt = torch.optim.AdamW([dict(params=tp, lr=args.lr), dict(params=head.parameters(), lr=args.lr * 10)], weight_decay=0.05)
    base = [g_['lr'] for g_ in opt.param_groups]
    total = args.epochs * len(loader)
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(int(0.05 * total), 1)
    use_amp = dev == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    step, t0, losses = 0, time.time(), []
    model.train()
    while step < total:
        for xb, yb in loader:
            if step >= total:
                break
            lr_f = (step + 1) / warm if step < warm else 0.5 * (1 + np.cos(np.pi * (step - warm) / max(total - warm, 1)))
            for g_, b0 in zip(opt.param_groups, base):
                g_['lr'] = b0 * lr_f
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = head(model((xb * mask - mean) / std))
            loss = F.binary_cross_entropy_with_logits(out.float(), yb, pos_weight=pw)
            if not torch.isfinite(loss):
                sys.exit(f'{BAD} loss is {loss.item()} at step {step}')
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            step += 1
            if step % 40 == 0 or step == 1 or step == total:
                say(OK, f'step {step}/{total} loss {np.mean(losses[-20:]):.3f} ({time.time()-t0:.0f}s)')
    model.eval()
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, os.path.join(CK, f'{run}.pt'))
    F1 = T.extract(model, X, mean, std, mask, dev)
    if not np.isfinite(F1).all():
        sys.exit(f'{BAD} features contain NaN/inf')
    np.save(os.path.join(FE, f'{run}.npy'), F1.astype(np.float16))
    res = dict(run=run, variant='sup', hold=args.hold, seed=args.seed, params_trainable=int(sum(p.numel() for p in tp)),
               train_seconds=time.time() - t0, loss_first=float(np.mean(losses[:5])), loss_last=float(np.mean(losses[-5:])))
    res.update(T.evaluate(F1, meta, args.hold))
    json.dump(res, open(os.path.join(R, f'{run}.json'), 'w'), indent=1)
    T.summary(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['sup', 'rare', 'ood2', 'loco'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--hold', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--boot', type=int, default=500)
    ap.add_argument('--support_sets', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--k_blocks', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    print(f'WORK={T.WORK}\n')
    if a.stage == 'ood2' and not a.hold:
        a.hold = 'cam1'
    {'sup': sup_stage, 'rare': rare_stage, 'ood2': ood2_stage, 'loco': loco_stage}[a.stage](a)
