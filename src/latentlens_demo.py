#!/usr/bin/env python
"""Latent Lens demo: upload a retinal photo -> similar known cases, calibrated disease scores, image-quality flag,
unfamiliar-image flag. Research prototype, NOT a medical device. Needs latentlens_train.py in the same folder.

  !python latentlens_demo.py --stage build  --run b3_fmix_s0     # one-off: gallery, thumbnails, probes, thresholds (no GPU needed)
  !python latentlens_demo.py --stage test   --run b3_fmix_s0     # self-check: model reloads and matches the saved features
  !python latentlens_demo.py --stage launch --run b3_fmix_s0     # web app (set --share for a public link)

Every number shown comes from the real runs: probes are fit on the RFMiD training images only; the temperature is fitted on
the validation images; the per-class AUROC column is measured on held-out images (test, or val+test for rare classes).
"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
import latentlens_train as T
from latentlens_train import say, OK, BAD, WARN, l2

ASSETS = os.path.join(T.WORK, 'demo_assets')
os.makedirs(ASSETS, exist_ok=True)
DISCLAIMER = ('Research prototype for a hackathon. Not a medical device, not validated for clinical use. '
              'Scores come from linear probes on embeddings fitted to the public RFMiD data; do not use them to make care decisions.')


# ----------------------------------------------------------------------------- preprocessing (same as the data cache)
def preprocess(im_rgb, size=224):
    """Crop the black border, pad to a square, resize to 224; also returns simple quality statistics."""
    import cv2
    im = np.ascontiguousarray(im_rgb[:, :, :3]).astype(np.uint8)
    h, w = im.shape[:2]
    g = im.max(axis=2)
    ys, xs = np.where(g[::8, ::8] > 20)
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
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    m = g > 20
    px = im[m] if m.any() else im.reshape(-1, 3)
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    stats = dict(contrast=float(px.std(0).mean()), sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return out, stats


def jpeg(x, q=85):
    import cv2
    return cv2.imencode('.jpg', cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


# ----------------------------------------------------------------------------- build assets (numpy / sklearn only)
def build(args):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.covariance import LedoitWolf
    from scipy.optimize import minimize_scalar
    cdir = T.find_cache()
    meta = pd.read_csv(os.path.join(cdir, 'meta.csv'))
    X = np.load(os.path.join(cdir, 'images_224.npy'), mmap_mode='r')
    ff = T.collect('features', '.npy').get(args.run)
    if ff is None:
        sys.exit(f'{BAD} features for {args.run} not found. Attach the notebook outputs that trained it.')
    F = l2(np.load(ff).astype(np.float32))
    assert len(F) == len(meta) == len(X), 'cache / features / meta do not line up'
    rfm = meta.dataset.values == 'rfmid'
    rf = meta[rfm].reset_index(drop=True)
    cols = T.label_cols(rf)
    Y = rf[cols].fillna(0).values.astype(int)
    Fr = F[rfm]
    tr, va, te = [(rf.split == s).values for s in ('train', 'val', 'test')]
    pool = va | te
    ntr = Y[tr].sum(0)
    sig = lambda z: 1 / (1 + np.exp(-z))
    keep, W, B, MU, SC, TEMP, AUC, NPOS = [], [], [], [], [], [], [], []
    from sklearn.metrics import roc_auc_score
    for j, c in enumerate(cols):
        if ntr[j] < 8 or Y[pool, j].sum() < 4:
            continue
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Fr[tr], Y[tr, j])
        zv = pipe.decision_function(Fr[va])
        t = 1.0
        if Y[va, j].sum() >= 5:
            def nll(tt):
                p = np.clip(sig(zv / tt), 1e-7, 1 - 1e-7)
                return -np.mean(Y[va, j] * np.log(p) + (1 - Y[va, j]) * np.log(1 - p))
            t = float(minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x)
        ev = te if ntr[j] >= 30 else pool                    # rare classes are scored on val+test pooled
        auc = float(roc_auc_score(Y[ev, j], pipe.decision_function(Fr[ev]))) if 0 < Y[ev, j].sum() < ev.sum() else float('nan')
        sc, lr = pipe.named_steps['standardscaler'], pipe.named_steps['logisticregression']
        keep.append(j); MU.append(sc.mean_); SC.append(sc.scale_); W.append(lr.coef_[0]); B.append(lr.intercept_[0])
        TEMP.append(t); AUC.append(auc); NPOS.append(int(ntr[j]))
        say(OK, f'probe {c:13s} train positives {ntr[j]:4d}  held-out AUROC {auc:.3f}  temperature {t:.2f}')
    # unfamiliar-image score: Mahalanobis distance to the RFMiD training distribution
    lw = LedoitWolf().fit(Fr[tr])
    maha = lambda Q: np.einsum('ij,jk,ik->i', Q - lw.location_, lw.precision_, Q - lw.location_)
    d_test = maha(Fr[te])
    thr = float(np.percentile(d_test, 95))
    # image-quality thresholds from the training images
    q_sharp = float(np.percentile(rf.loc[tr, 'sharp'].dropna(), 5)) if 'sharp' in rf else 0.0
    sd = [c for c in ('std_r', 'std_g', 'std_b') if c in rf.columns]
    q_cont = float(np.percentile(rf.loc[tr, sd].mean(axis=1).dropna(), 5)) if sd else 0.0
    # case library = RFMiD training images; thumbnails as small JPEGs
    lib = np.where(rfm)[0][tr]
    thumbs = [jpeg(np.asarray(X[i])[::2, ::2]) for i in lib]
    cam = rf.camera.values[tr]
    ex = []
    for c, k in (('cam0', 4), ('cam1', 4), ('cam2', 2)):
        ex += list(np.where(rfm)[0][(rf.camera.values == c) & te][:k])
    ex += list(np.where(meta.dataset.values == 'idrid')[0][:2])
    np.savez_compressed(os.path.join(ASSETS, 'probes.npz'), cls=np.array([cols[j] for j in keep]), mu=np.array(MU), sc=np.array(SC),
                        w=np.array(W), b=np.array(B), temp=np.array(TEMP), auc=np.array(AUC), npos=np.array(NPOS),
                        maha_loc=lw.location_, maha_prec=lw.precision_, maha_thr=thr, q_sharp=q_sharp, q_cont=q_cont,
                        d_test_sorted=np.sort(d_test))
    labs_txt = ['; '.join(cols[k] for k in np.where(y)[0] if cols[k] != 'Disease_Risk') or 'no disease label' for y in Y[tr]]
    np.savez_compressed(os.path.join(ASSETS, 'gallery.npz'), feats=Fr[tr].astype(np.float16),
                        cam=np.array([str(c) for c in cam], dtype='U8'), labels=np.array(labs_txt, dtype='U120'))
    pickle.dump(dict(thumbs=thumbs, examples=[(int(i), jpeg(np.asarray(X[i]))) for i in ex],
                     names=[meta.dataset.values[i] + '/' + str(meta.camera.values[i]) for i in ex]),
                open(os.path.join(ASSETS, 'images.pkl'), 'wb'))
    json.dump(dict(run=args.run, n_library=int(len(lib)), n_classes=len(keep), unfamiliar_threshold=thr,
                   quality_sharp_threshold=q_sharp, quality_contrast_threshold=q_cont),
              open(os.path.join(ASSETS, 'info.json'), 'w'), indent=1)
    say(OK, f'assets written to {ASSETS}: {len(keep)} probes, library of {len(lib)} cases, {len(ex)} examples')


# ----------------------------------------------------------------------------- analysis (numpy only)
class Assets:
    def __init__(self):
        p = np.load(os.path.join(ASSETS, 'probes.npz'), allow_pickle=False)
        self.p = {k: p[k] for k in p.files}
        g = np.load(os.path.join(ASSETS, 'gallery.npz'), allow_pickle=False)
        self.lib_f, self.lib_cam, self.lib_lab = g['feats'].astype(np.float32), g['cam'], g['labels']
        im = pickle.load(open(os.path.join(ASSETS, 'images.pkl'), 'rb'))
        self.thumbs, self.examples, self.ex_names = im['thumbs'], im['examples'], im['names']
        self.info = json.load(open(os.path.join(ASSETS, 'info.json')))


def analyse_feature(A, f, stats, k=5):
    """f: 384-d embedding of the processed image. Returns plain python structures (testable without torch)."""
    P = A.p
    f = f / (np.linalg.norm(f) + 1e-8)
    z = (((f[None] - P['mu']) / P['sc']) * P['w']).sum(1) + P['b']
    prob = 1 / (1 + np.exp(-z / P['temp']))
    order = np.argsort(-prob)
    rows = [dict(condition=str(P['cls'][i]), probability=float(prob[i]), held_out_auc=float(P['auc'][i]), train_positives=int(P['npos'][i]))
            for i in order]
    d = f - P['maha_loc']
    dist = float(d @ P['maha_prec'] @ d)
    pct = float((P['d_test_sorted'] < dist).mean() * 100)
    sims = A.lib_f @ f
    top = np.argsort(-sims)[:k]
    quality = []
    if stats['sharp'] < float(P['q_sharp']):
        quality.append('low sharpness (blurry or out of focus)')
    if stats['contrast'] < float(P['q_cont']):
        quality.append('low contrast or very dark')
    return dict(rows=rows, unfamiliar=bool(dist > float(P['maha_thr'])), distance=dist, percentile=pct,
                quality=quality, similar=[dict(index=int(i), similarity=float(sims[i]), camera=str(A.lib_cam[i]), labels=str(A.lib_lab[i])) for i in top])


# ----------------------------------------------------------------------------- model + app (torch, gradio)
class Embedder:
    def __init__(self, run):
        import torch
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T.make_model(self.dev)
        self.mean, self.std = T.norm_stats(self.model, self.dev)
        ck = T.collect('ckpt', '.pt').get(run)
        if ck is None:
            sys.exit(f'{BAD} checkpoint {run}.pt not found. Attach the notebook output that trained it.')
        r = self.model.load_state_dict(torch.load(ck, map_location='cpu'), strict=False)
        if r.unexpected_keys:
            sys.exit(f'{BAD} unexpected keys in checkpoint: {r.unexpected_keys[:3]}')
        self.model.eval()
        self.mask = torch.from_numpy(T.circle_mask()).to(self.dev).view(1, 1, T.SIZE, T.SIZE)

    def __call__(self, img224):
        return T.extract(self.model, img224[None], self.mean, self.std, self.mask, self.dev)[0]


def selftest(args):
    meta = pd.read_csv(os.path.join(T.find_cache(), 'meta.csv'))
    X = np.load(os.path.join(T.find_cache(), 'images_224.npy'), mmap_mode='r')
    F = l2(np.load(T.collect('features', '.npy')[args.run]).astype(np.float32))
    emb, A = Embedder(args.run), Assets()
    idx = [int(i) for i, _ in A.examples[:3]]
    worst = 0.0
    for i in idx:
        f = emb(np.asarray(X[i]))
        worst = max(worst, float(np.abs(l2(f[None])[0] - F[i]).max()))
    say(OK if worst < 0.05 else BAD, f'model reloaded from the checkpoint: max abs feature difference to the saved features {worst:.4f}')
    # a made-up retinal-looking image must go through the full path without errors
    rng = np.random.RandomState(0)
    fake = (rng.rand(600, 800, 3) * 255).astype(np.uint8)
    img, st = preprocess(fake)
    out = analyse_feature(A, emb(img), st)
    say(OK, f'full path ok: top condition {out["rows"][0]["condition"]} p={out["rows"][0]["probability"]:.2f}, '
            f'unfamiliar={out["unfamiliar"]}, quality flags={out["quality"]}')
    for i in idx:
        o = analyse_feature(A, emb(np.asarray(X[i])), dict(sharp=1e9, contrast=1e9))
        say(OK, f'example {i}: unfamiliar={o["unfamiliar"]} (percentile {o["percentile"]:.0f}), top similar labels: {o["similar"][0]["labels"]}')


def launch(args):
    import cv2
    import gradio as gr
    emb, A = Embedder(args.run), Assets()

    def run(img):
        if img is None:
            return None, 'Upload a retinal photograph or pick an example.', [], []
        proc, st = preprocess(img)
        o = analyse_feature(A, emb(proc), st)
        flags = []
        flags.append('Image quality: ' + ('OK' if not o['quality'] else 'WARNING - ' + ', '.join(o['quality'])))
        flags.append(f'Familiarity: ' + ('UNFAMILIAR - this image lies far outside the images the model was built on '
                                       f'(farther than 95% of known test images; its distance is higher than {o["percentile"]:.0f}% of them).'
                                       if o['unfamiliar'] else f'within the range of known images (distance higher than {o["percentile"]:.0f}% of known test images).'))
        flags.append('The unfamiliar-image flag is a rough screening aid (about 0.7 AUROC for unseen cameras in our tests).')
        table = [[r['condition'], f'{100 * r["probability"]:.1f}%', f'{r["held_out_auc"]:.2f}', r['train_positives']] for r in o['rows'][:8]]
        gal = []
        for s in o['similar']:
            th = cv2.cvtColor(cv2.imdecode(np.frombuffer(A.thumbs[s['index']], np.uint8), 1), cv2.COLOR_BGR2RGB)
            gal.append((th, f'{s["labels"]} | {s["camera"]} | similarity {s["similarity"]:.2f}'))
        return proc, '\n'.join(flags), table, gal

    ex_imgs = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB) for _, b in A.examples]
    with gr.Blocks(title='Latent Lens') as demo:
        gr.Markdown('# Latent Lens\nCamera-robust retinal image embeddings: similar known cases, calibrated disease scores, image-quality and '
                    'unfamiliar-image flags.\n\n**' + DISCLAIMER + '**')
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label='Retinal photograph', type='numpy')
                sel = gr.Dropdown(choices=[f'{i}: {n}' for i, n in enumerate(A.ex_names)], value=None,
                                  label='Or load an example (public test image: dataset/camera)')
                sel.change(lambda v: None if v is None else ex_imgs[int(v.split(':')[0])], sel, inp)
                btn = gr.Button('Analyse', variant='primary')
            with gr.Column():
                proc = gr.Image(label='Image as the model sees it (224 x 224)')
                flags = gr.Textbox(label='Flags', lines=5)
        table = gr.Dataframe(headers=['Condition', 'Calibrated probability', 'Held-out AUROC of this score', 'Training positives'],
                             label='Top scores (probability is calibrated on validation images; rare conditions have few examples, so read the AUROC and counts)')
        gal = gr.Gallery(label='Most similar known cases (RFMiD training images): disease labels | camera | similarity', columns=5, height=260)
        btn.click(run, inp, [proc, flags, table, gal])
    demo.launch(share=args.share, debug=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['build', 'test', 'launch'])
    ap.add_argument('--run', default='b3_fmix_s0')
    ap.add_argument('--share', action='store_true')
    a = ap.parse_args()
    print(f'WORK={T.WORK}\n')
    {'build': build, 'test': selftest, 'launch': launch}[a.stage](a)
