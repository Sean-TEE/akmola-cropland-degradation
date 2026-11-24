#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-registered check of the shrunk baseline scale against the per-pixel and cell-pooled scales.

Usage: python stepb.py DATA_DIR OUT_DIR VARIANT [quick]
VARIANT: V0 (per-pixel scale, control), V1 (median scale of the 25 km cell), V2 (pixel scale shrunk towards the cell; primary).
DATA_DIR must contain mittcda_sample_v4_sigma.csv, era5land_covariates_2013_2026.tif and the files read by mittcda_v3_analysis.load()."""
import sys, os, json, time
import numpy as np, pandas as pd, rasterio
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))      # mittcda_v3_analysis.py may sit one level up
from pyproj import Transformer
import mittcda_v3_analysis as M

IDX = M.IDX
DELTA = np.array([-1.0, -1.0, 1.0, 1.0, 1.0]); B = np.array([0.173, 0.178, -0.370, -0.211, 0.0])
FLOOR = np.array([0.01, 0.01, 0.3, 0.005, 0.002])                     # NDVI, NDMI, LST, TGSI, SI
V_EST = {7: 0.2957, 6: 0.2900, 5: 0.5004}                             # variance of ln(1.4826 MAD) for 7, 6, 5 Gaussian values
YRS = list(range(2013, 2027)); MINC = 30

def load(D):
    df = pd.read_csv(D + '/mittcda_sample_v4_sigma.csv'); n = len(df)
    A = np.stack([df[[f'{k}_{y}' for y in YRS]].values for k in IDX], 1)                      # (n,5,14) adjusted anomalies, pixel scale
    sig = df[['sigma_' + k for k in IDX]].values; raw = df[['sigraw_' + k for k in IDX]].values; nb = df[['nbase_' + k for k in IDX]].values
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:32642', always_xy=True).transform(df.lon.values, df.lat.values)
    cell = np.floor(x / M.CELL).astype(int) * 10000 + np.floor(y / M.CELL).astype(int)
    r = rasterio.open(D + '/era5land_covariates_2013_2026.tif'); E = r.read().astype(float); names = list(r.descriptions)
    Pc = np.stack([E[names.index(f'P_{yy}')] for yy in YRS], 0)                               # (14,rows,cols)
    mu = np.median(Pc[:7], 0); sg = np.maximum(1.4826 * np.median(np.abs(Pc[:7] - mu), 0), 1e-6); ZPc = (Pc - mu) / sg
    col = np.floor((df.lon.values - r.transform.c) / r.transform.a).astype(int); row = np.floor((df.lat.values - r.transform.f) / r.transform.e).astype(int)
    ZP = ZPc[:, row, col].T                                                                    # (n,14)
    return df, A, sig, raw, nb, cell, ZP, x, y

class Cells:
    """Median of a quantity over the sample pixels of a 25 km cell; cells with fewer than MINC valid values take the median of the sample."""
    def __init__(self, cell):
        self.codes, _ = pd.factorize(cell)
    def median(self, V):                                                                        # V (n,k) -> (n,k)
        d = pd.DataFrame(V); g = d.groupby(self.codes); med = g.transform('median').values; cnt = g.transform('count').values
        return np.where(cnt >= MINC, med, np.nanmedian(V, 0)[None, :])
    def within_var(self, V):                                                                    # pooled within-cell variance per column
        d = pd.DataFrame(V); g = d.groupby(self.codes); dev = V - g.transform('mean').values; cnt = g.transform('count').values
        ok = cnt >= MINC; return np.array([np.nanvar(dev[ok[:, i], i]) for i in range(V.shape[1])])

IU = np.triu_indices(12, 1)
def fast_Rk(X):
    """Per-axis trend flags; with tau >= critical value > 0 and no ties the Theil-Sen slope is positive as well."""
    d = X[..., IU[1]] - X[..., IU[0]]; S = np.nansum(np.sign(d), -1); nv = (~np.isnan(X)).sum(-1); npair = nv * (nv - 1) / 2
    return np.where(npair > 0, S / np.maximum(npair, 1), np.nan) >= M.crit_by_n(nv)

def restd(X, w, C):
    """Standardization of the variant on the first seven positions: median of the pixel; scale = MAD of the pixel shrunk towards the cell with weight w."""
    b = X[:, :, :7]; mu = np.nanmedian(b, -1, keepdims=True); s = np.maximum(1.4826 * np.nanmedian(np.abs(b - mu), -1), 1e-9)
    if w is not None:
        s = np.exp(w * np.log(s) + (1 - w) * np.log(C.median(s)))
    return (X - mu) / s[:, :, None]

def evaluate(X, w, C):
    Z = restd(X, w, C); T = M.signature(M.active(Z[:, :, :12])); Rk = fast_Rk(X[:, :, :12]); R = Rk.any(1)
    ex = np.nan_to_num(Z[:, :, 12:], nan=-99) > 1.5; ok = ~np.isnan(Z[:, :, 12:])
    t, r_, i = T.mean(), R.mean(), (T & R).mean()
    o = {'T': 100 * t, 'R': 100 * r_, 'J': i / (t + r_ - i) if (t + r_ - i) > 0 else np.nan}
    g = {'T and R': T & R, 'T only': T & ~R, 'R only': R & ~T, 'none': ~T & ~R, 'T': T, 'R': R}
    for k, m in g.items():
        for j in (0, 1):
            v = m & ok[:, 0, j]; o['%s|%d' % (k, j)] = 100 * ex[v, 0, j].mean() if v.any() else np.nan
    for j in (0, 1):
        own = (ex[:, :, j] & Rk & ok[:, :, j]).any(1); valid = (Rk & ok[:, :, j]).any(1); m = g['R only'] & valid
        o['R only own axis|%d' % j] = 100 * own[m].mean() if m.any() else np.nan
        for k in ('T and R', 'T only', 'R only', 'T', 'R', 'R only own axis'):
            o['%s-none|%d' % (k, j)] = o['%s|%d' % (k, j)] - o['none|%d' % j]
    return o

def summarize(rows, obs):
    d = pd.DataFrame(rows); out = {}
    for k in d.columns:
        v = d[k].dropna().values
        out[k] = dict(mean=float(v.mean()), sd=float(v.std()), q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975)), max=float(v.max()),
                      p=float((1 + (v >= obs[k]).sum()) / (len(v) + 1)) if k in obs and not np.isnan(obs[k]) else None)
    return out

def main():
    D, O, VAR = sys.argv[1], sys.argv[2], sys.argv[3]; quick = 'quick' in sys.argv; t0 = time.time()
    df, A, sig, raw, nb, cell, ZP, x, y = load(D); n = len(df); C = Cells(cell); res = {'variant': VAR, 'n': n}
    # ---- recover the standardized values and check the reconstruction
    Z = DELTA[None, :, None] * A + B[None, :, None] * ZP[:, None, :]
    bm = np.nanmedian(Z[:, :, :7], -1); bs = 1.4826 * np.nanmedian(np.abs(Z[:, :, :7] - bm[:, :, None]), -1)
    res['check'] = dict(zp_mean_by_year=ZP.mean(0).round(2).tolist(),
                        abs_baseline_median_q50_q99={k: [round(float(np.nanquantile(np.abs(bm[:, i]), q)), 4) for q in (.5, .99)] for i, k in enumerate(IDX)},
                        baseline_scale_vs_ratio_maxdiff_q99={k: round(float(np.nanquantile(np.abs(bs[:, i] - raw[:, i] / sig[:, i]), .99)), 4) for i, k in enumerate(IDX)})
    # ---- scale of the variant
    lnr = np.log(np.maximum(raw, 1e-12)); sc = C.median(raw); vobs = C.within_var(np.where(raw > 0, lnr, np.nan))
    vest = np.full(nb.shape, np.inf)
    for k_, v_ in V_EST.items(): vest[nb == k_] = v_
    w2 = np.clip(1 - vest / vobs[None, :], 0, 1)
    if VAR == 'V0': st = sig.copy(); w = None
    elif VAR == 'V1': st = np.maximum(sc, FLOOR[None, :]); w = np.zeros_like(raw)
    else: st = np.maximum(np.exp(w2 * lnr + (1 - w2) * np.log(sc)), FLOOR[None, :]); w = w2
    res['scale'] = dict(within_cell_var_ln_sigma={k: round(float(vobs[i]), 3) for i, k in enumerate(IDX)},
                        weight_7yr={k: round(float(max(0, 1 - V_EST[7] / vobs[i])), 3) for i, k in enumerate(IDX)},
                        ratio_q05_q50_q95={k: [round(float(v), 2) for v in np.nanquantile(sig[:, i] / st[:, i], [.05, .5, .95])] for i, k in enumerate(IDX)})
    Av = DELTA[None, :, None] * (Z * (sig / st)[:, :, None] - B[None, :, None] * ZP[:, None, :])
    V = np.stack([Av[:, :, j] - C.median(Av[:, :, j]) for j in range(14)], 2)                 # relative anomalies of the variant, 14 years
    if VAR == 'V0':
        res['check']['V0_equals_A'] = float(np.nanmax(np.abs(Av - A)))
    # ---- layers of the variant (scale as in the stored product: no re-standardization)
    I = M.active(V[:, :, :12]); T = M.signature(I); Rk, Imp = M.trend_flags(V[:, :, :12]); R = Rk.any(1); assert (Rk == fast_Rk(V[:, :, :12])).all()
    s = M.summ(T, R); mc = M.mech_code(I)
    ex = np.nan_to_num(V[:, :, 12:], nan=-99) > 1.5; ok = ~np.isnan(V[:, :, 12:])
    g = {'T and R': T & R, 'T only': T & ~R, 'R only': R & ~T, 'none': ~T & ~R, 'T': T, 'R': R}
    res['layers'] = dict(T=s['T'], R=s['R'], TR=s['TR'], J=s['J'], improving_any=100 * float(Imp.any(1).mean()), any_axis=100 * float(I.any(1).mean()),
                         mech={f'D{d}': 100 * float((mc == c).mean()) for d, c in ((1, 1), (2, 4), (3, 2), (4, 3))},
                         exceed_NDVI_by_year=[round(100 * float(np.mean(V[:, 0, j][~np.isnan(V[:, 0, j])] > 1.5)), 1) for j in range(14)],
                         spread_NDVI_by_year=[round(float(1.4826 * np.nanmedian(np.abs(V[:, 0, j] - np.nanmedian(V[:, 0, j])))), 2) for j in range(14)],
                         n={k: int(m.sum()) for k, m in g.items()},
                         holdout_exceed={k: [round(100 * float(ex[m & ok[:, 0, j], 0, j].mean()), 1) for j in (0, 1)] for k, m in g.items()},
                         holdout_median={k: [round(float(np.nanmedian(V[m, 0, 12 + j])), 2) for j in (0, 1)] for k, m in g.items()},
                         trajectory_NDVI={k: np.nanmedian(V[m, 0, :], 0).round(2).tolist() for k, m in g.items() if k in ('T and R', 'T only', 'R only', 'none')})
    # gain: slope of the adjusted anomaly of the pixel on the median of its cell over 2013-2024
    Xr = np.stack([C.median(Av[:, 0, j][:, None])[:, 0] for j in range(12)], 1); Y = Av[:, 0, :12]; okk = ~np.isnan(Y); Xc = np.where(okk, Xr, np.nan)
    gain = np.nansum((Xc - np.nanmean(Xc, 1, keepdims=True)) * (Y - np.nanmean(Y, 1, keepdims=True)), 1) / np.nansum((Xc - np.nanmean(Xc, 1, keepdims=True)) ** 2, 1)
    res['layers']['gain_NDVI'] = {k: round(float(np.nanmedian(gain[m])), 2) for k, m in g.items()}
    # share of T made of pixels whose own scale was underestimated relative to the cell
    rel_scale = raw / sc
    res['layers']['pixel_to_cell_scale_median'] = {k: {a: round(float(np.nanmedian(rel_scale[m, i])), 2) for i, a in enumerate(IDX)} for k, m in g.items()}
    # ---- like-for-like observation and scenarios
    obs = evaluate(V, w, C); res['observed'] = obs
    reps = dict(p12=6, w12=20, p14=8, w14=30, gauss=3) if quick else dict(p12=40, w12=200, p14=60, w14=400, gauss=20)
    rng = np.random.default_rng({'V0': 500, 'V1': 501, 'V2': 502}[VAR])
    def scenario(L, whole, k):
        rows = []
        for _ in range(k):
            if whole: p = np.argsort(rng.random(L)); Xp = V[:, :, :L][:, :, p]
            else: p = np.argsort(rng.random((n, L)), 1); Xp = np.take_along_axis(V[:, :, :L], p[:, None, :], 2)
            if L == 12: Xp = np.concatenate([Xp, np.full((n, 5, 2), np.nan)], 2)
            rows.append(evaluate(Xp, w, C))
        return rows
    obs12 = evaluate(np.concatenate([V[:, :, :12], np.full((n, 5, 2), np.nan)], 2), w, C)
    for lab, L, whole, k in (('pixel_12', 12, False, reps['p12']), ('whole_12', 12, True, reps['w12']), ('pixel_14', 14, False, reps['p14']), ('whole_14', 14, True, reps['w14'])):
        rows = scenario(L, whole, k); keys = ['T', 'R', 'J'] if L == 12 else list(rows[0].keys())
        res[lab] = summarize([{kk: r_[kk] for kk in keys} for r_ in rows], obs12 if L == 12 else obs); res[lab]['reps'] = k
        print(VAR, lab, 'done, %.0f s' % (time.time() - t0), flush=True)
        json.dump(res, open(O + '/stepb_%s.json' % VAR, 'w'), indent=1, default=float)
    # Gaussian noise with the correlation of Table 3 and the missing-value pattern of the sample
    Cm = np.array([[1, .72, .31, .37, .63], [.72, 1, .31, .12, .33], [.31, .31, 1, .16, .27], [.37, .12, .16, 1, .36], [.63, .33, .27, .36, 1]]); Lc = np.linalg.cholesky(Cm)
    nanmask = np.isnan(V); rows = []
    for _ in range(reps['gauss']):
        G = np.einsum('ab,nbt->nat', Lc, rng.standard_normal((n, 5, 14))); G[nanmask] = np.nan; rows.append(evaluate(G, w, C))
    res['gaussian'] = pd.DataFrame(rows).mean().to_dict()
    # ---- decision rule (fixed in advance)
    c1 = [res['whole_14']['T-none|%d' % j]['p'] for j in (0, 1)]; c2 = res['pixel_12']['T']['mean'] / obs12['T']
    res['decision'] = dict(p_T_minus_none_2025=c1[0], p_T_minus_none_2026=c1[1], condition1=bool(c1[0] < 0.05 and c1[1] < 0.05),
                           noise_share_of_T=c2, condition2=bool(c2 < 0.5), proceed=bool(c1[0] < 0.05 and c1[1] < 0.05 and c2 < 0.5))
    json.dump(res, open(O + '/stepb_%s.json' % VAR, 'w'), indent=1, default=float)
    print(VAR, 'done in %.0f s' % (time.time() - t0), json.dumps(res['decision']))

if __name__ == '__main__':
    main()
