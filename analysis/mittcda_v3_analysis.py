#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MITT-CDA v3: all sample-based tables of the manuscript.

Usage:  python mittcda_v3_analysis.py DATA_DIR OUT_DIR [--fast] [--supplement]

DATA_DIR must contain
  mittcda_sample_v3_holdout.csv          random sample of cropland pixels (GEE script 04)
  ref_grid_25km_v3_<AXIS>.csv            25 km cell medians of the adjusted anomalies (GEE script 02, stage A)
  diag_regional_median_A_v3.csv          oblast-wide medians (GEE script 01)
  era5land_covariates_2013_2026.tif      ERA5-Land covariates on the 0.1 degree grid (GEE script 04)
  v3_areas_grid_r2.csv, v3_areas_by_district_grid_r2.csv   full-raster areas (GEE script 02, stage C)
  diag_n_valid_years_by_axis.csv, diag_T_R_by_n_min.csv, sigma_floor_area.csv, composite_doy_by_year.csv

Output: OUT_DIR/results.json. All random procedures use fixed seeds.
Requires numpy, pandas, scipy, pyproj, rasterio.
"""
import os, sys, json, time, warnings
from math import factorial
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from pyproj import Transformer
import rasterio

warnings.filterwarnings('ignore')
IDX = ['NDVI', 'NDMI', 'LST', 'TGSI', 'SI']
YEARS = list(range(2013, 2025)); HOLD = [2025, 2026]
AREA = 57663.0755                                   # km2, analysis area (full raster)
W = np.array([0.200, 0.218, 0.212, 0.194, 0.176])   # PCA weights of the composite index
CELL = 25000.0                                      # m, reference grid (EPSG:32642)
MIN_PIX = 10000
ADM = {37163: 'Akkol', 37164: 'Arshaly', 37165: 'Astrakhanskiy', 37166: 'Atbasar', 37167: 'Bulandy',
       37168: 'Egindykol', 37169: 'Ereymentau', 37170: 'Esil', 37171: 'Korghalzhyn', 37172: 'Sandyktau',
       37173: 'Shortandy', 37174: 'Tselinograd', 37175: 'Zhaksy', 37176: 'Zharkaiyn'}

# ------------------------------------------------------------------ exact Kendall null distribution
def inv_pmf(n):
    d = [1]
    for m in range(2, n + 1):
        new = [0] * (len(d) + m - 1)
        for i, v in enumerate(d):
            for j in range(m):
                new[i + j] += v
        d = new
    return np.array(d, float) / factorial(n)

def crit_tau(n, alpha2):
    """Smallest attainable tau with exact two-sided p < alpha2 (no ties)."""
    N = n * (n - 1) // 2; c = np.cumsum(inv_pmf(n))
    return (N - 2 * np.where(2 * c < alpha2)[0].max()) / N

CRIT = {a: {n: crit_tau(n, a) for n in range(8, 13)} for a in (0.10, 0.05, 0.01)}
PGE = {n: np.cumsum(inv_pmf(n)) for n in range(8, 13)}        # PGE[n][inv] = P(S' >= N - 2 inv)

IU = np.triu_indices(12, 1); DT = (IU[1] - IU[0]).astype(float)

def kendall(X):
    """S, tau-a, Theil-Sen slope and number of valid years along the last axis (NaN-aware)."""
    d = X[..., IU[1]] - X[..., IU[0]]
    S = np.nansum(np.sign(d), -1); nv = (~np.isnan(X)).sum(-1); npair = nv * (nv - 1) / 2
    tau = np.where(npair > 0, S / np.maximum(npair, 1), np.nan)
    return S, tau, np.nanmedian(d / DT, -1), nv

def crit_by_n(nv, alpha2=0.05, fixed=None):
    c = np.full(nv.shape, 9.0)
    if fixed is not None:
        c[:] = fixed; return c
    for n, v in CRIT[alpha2].items():
        c[nv == n] = v - 1e-9
    return c

def trend_flags(X, alpha2=0.05, fixed=None):
    S, tau, slope, nv = kendall(X)
    c = crit_by_n(nv, alpha2, fixed)
    return (slope > 0) & (tau >= c), (slope < 0) & (tau <= -c)

def active(X, thr=1.5, k=3):
    return (np.nan_to_num(X[..., 7:12], nan=-99) > thr).sum(-1) >= k

def signature(I):
    nd, nm, ls, tg, si = (I[:, i] for i in range(5))
    return (nd & ls) | (nd & si) | (nd & tg)

def mech_code(I):
    nd, nm, ls, tg, si = (I[:, i] for i in range(5))
    m = np.zeros(len(I), int); m[nd & ls] = 1; m[nd & si] = 2; m[nd & tg] = 3; m[nd & nm & ls] = 4
    return m

def restd(X):
    b = X[:, :, :7]; mu = np.nanmedian(b, -1, keepdims=True)
    sg = 1.4826 * np.nanmedian(np.abs(b - mu), -1, keepdims=True)
    return (X - mu) / np.maximum(sg, 1e-9)

def summ(T, R, m=None):
    if m is not None:
        T, R = T[m], R[m]
    t, r, i = T.mean(), R.mean(), (T & R).mean(); u = t + r - i
    return dict(n=int(len(T)), T=100 * t, R=100 * r, TR=100 * i, J=i / u if u else float('nan'),
                R_in_T=i / r if r else float('nan'), T_in_R=i / t if t else float('nan'),
                T_km2=t * AREA, R_km2=r * AREA, TR_km2=i * AREA)

# ------------------------------------------------------------------ data
def load(D):
    f = D + '/mittcda_sample_v3_holdout.csv'
    if not os.path.exists(f):                       # the v4 export holds the same rows and columns plus the baseline statistics
        f = D + '/mittcda_sample_v4_sigma.csv'
    df = pd.read_csv(f)
    A = np.stack([df[[f'{k}_{y}' for y in YEARS]].values for k in IDX], 1)
    H = np.stack([np.stack([df[f'{k}_{y}'].values for y in HOLD], 1) for k in IDX], 1)      # (n,5,2)
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:32642', always_xy=True).transform(df.lon.values, df.lat.values)
    reg = pd.read_csv(D + '/diag_regional_median_A_v3.csv').iloc[0]
    ref_reg = np.array([[reg[f'{k}_{yr}'] for yr in YEARS] for k in IDX])
    # 25 km reference: bilinear interpolation between cell centres; cells below MIN_PIX -> regional median
    n = len(df); fx = x / CELL - 0.5; fy = y / CELL - 0.5
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int); wx = fx - ix; wy = fy - iy
    ref = np.empty_like(A); ncell = {}
    for ai, k in enumerate(IDX):
        t = pd.read_csv(D + f'/ref_grid_25km_v3_{k}.csv'); ncell[k] = (len(t), int((t[k + '_2020_count'] >= MIN_PIX).sum()))
        t = t[t[k + '_2020_count'] >= MIN_PIX]
        xy = t['.geo'].map(lambda s: json.loads(s)['coordinates'][0][0])
        cx = (xy.map(lambda p: p[0]) / CELL).round().astype(int).values; cy = (xy.map(lambda p: p[1]) / CELL).round().astype(int).values
        for j, yr in enumerate(YEARS):
            lut = dict(zip(zip(cx, cy), t[f'{k}_{yr}_median'].values))
            num = np.zeros(n); den = np.zeros(n)
            for dx, dy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
                v = np.array([lut.get(key, np.nan) for key in zip(ix + dx, iy + dy)]); ok = ~np.isnan(v)
                num[ok] += w[ok] * v[ok]; den[ok] += w[ok]
            ref[:, ai, j] = np.where(den > 0, num / np.maximum(den, 1e-12), ref_reg[ai, j])
    return df, A, H, x, y, ref_reg, ref, ncell

# ------------------------------------------------------------------ supplementary tables (no shared random stream)
def supplement(res, df, A, H, x, y, ref_reg, ref):
    As = A - ref
    T3 = signature(active(As)); Rk3, Imp3 = trend_flags(As); R3 = Rk3.any(1)
    Rk2, Imp2 = trend_flags(A)
    res['improving_any'] = dict(v3=100 * float(Imp3.any(1).mean()), v2=100 * float(Imp2.any(1).mean()),
                                v3_both_directions=100 * float((Imp3.any(1) & R3).mean()))
    # share of pixels above the anomaly threshold in each year, before and after removal of the local year effect
    def exc_year(X):
        return {k: [round(100 * float(np.mean(X[:, i, j][~np.isnan(X[:, i, j])] > 1.5)), 2) for j in range(12)] for i, k in enumerate(IDX)}
    res['exceed_by_year'] = dict(v2=exc_year(A), v3=exc_year(As))
    # share of the pixel-year variance of the anomalies removed by the references (values clamped to +-8 as in GEE)
    cl = lambda X: np.clip(X, -8, 8)
    res['variance_removed'] = {k: dict(regional=100 * float(1 - np.nanvar(cl(A[:, i] - ref_reg[i])) / np.nanvar(cl(A[:, i]))),
                                       grid=100 * float(1 - np.nanvar(cl(As[:, i])) / np.nanvar(cl(A[:, i])))) for i, k in enumerate(IDX)}
    # size of the reference cell: medians of the sample pixels of a square cell (>= 30 pixels, otherwise regional median)
    def sample_ref(size, minn=30):
        cid = np.floor(x / size).astype(int) * 100000 + np.floor(y / size).astype(int); out = np.empty_like(A)
        for i in range(5):
            for j in range(12):
                v = pd.Series(A[:, i, j]); gm = v.groupby(cid).transform('median'); gc = v.groupby(cid).transform('count')
                out[:, i, j] = np.where(gc >= minn, gm, np.nanmedian(A[:, i, j]))
        return out
    sc = {}
    for lab, size in (('100 km', 100000.0), ('50 km', 50000.0), ('25 km', 25000.0)):
        X = A - sample_ref(size); sc[lab] = summ(signature(active(X)), trend_flags(X)[0].any(1))
    res['reference_scale_sample'] = sc
    # hold-out years by layer, and spatial block bootstrap (50 km) of the exceedance shares of the four groups
    cell = np.floor(x / CELL).astype(int) * 10000 + np.floor(y / CELL).astype(int)
    def rel(v):
        s_ = pd.Series(v); gm = s_.groupby(cell).transform('median'); gc = s_.groupby(cell).transform('count')
        return (s_ - np.where(gc >= 30, gm, np.nanmedian(v))).values
    Hr = np.stack([rel(H[:, 0, j]) for j in range(2)], 1); ok = ~np.isnan(Hr); exc = np.nan_to_num(Hr, nan=-99) > 1.5
    lay = {'T': T3, 'not T': ~T3, 'R': R3, 'not R': ~R3}
    res['holdout_layers'] = {g: {str(yy): dict(n=int((m & ok[:, j]).sum()), rel_exc=100 * float(exc[m & ok[:, j], j].mean()),
                                               rel_med=float(np.nanmedian(Hr[m, j]))) for j, yy in enumerate(HOLD)} for g, m in lay.items()}
    g3 = {'none': ~T3 & ~R3, 'R only': R3 & ~T3, 'T only': T3 & ~R3, 'T and R': T3 & R3}
    bid = np.floor(x / 50000.0).astype(int) * 100000 + np.floor(y / 50000.0).astype(int); u, inv = np.unique(bid, return_inverse=True); nb = len(u)
    rng = np.random.default_rng(13); hb = {}
    for j, yy in enumerate(HOLD):
        num = {g: np.bincount(inv, weights=(m & ok[:, j] & exc[:, j]).astype(float), minlength=nb) for g, m in g3.items()}
        den = {g: np.bincount(inv, weights=(m & ok[:, j]).astype(float), minlength=nb) for g, m in g3.items()}
        vals = {g: [] for g in g3}; dif = {g: [] for g in g3}
        for _ in range(2000):
            s_ = rng.integers(0, nb, nb); sh = {g: 100 * num[g][s_].sum() / den[g][s_].sum() for g in g3}
            for g in g3:
                vals[g].append(sh[g]); dif[g].append(sh[g] - sh['none'])
        hb[str(yy)] = {g: dict(range=np.quantile(vals[g], [0.025, 0.975]).round(1).tolist(),
                               diff_from_none=np.quantile(dif[g], [0.025, 0.975]).round(1).tolist()) for g in g3}
    res['holdout_boot_50km'] = hb
    return res

def main():
    D, O = sys.argv[1], sys.argv[2]; fast = '--fast' in sys.argv
    REPS = 30 if fast else 200; BOOT = 200 if fast else 1000
    t0 = time.time(); res = {}
    df, A, H, x, y, ref_reg, ref, ncell = load(D)
    if '--supplement' in sys.argv:                  # add the supplementary tables to an existing results.json
        res = supplement(json.load(open(O + '/results.json', encoding='utf8')), df, A, H, x, y, ref_reg, ref)
        json.dump(res, open(O + '/results.json', 'w', encoding='utf8'), indent=1, default=float)
        print('supplement done in %.0f s' % (time.time() - t0)); return
    n = len(df); As = A - ref                       # A*: anomalies after removal of the local year effect
    nv_all = (~np.isnan(A)).sum(2); nmin = nv_all.min(1)
    tier = df.tier.values
    glc = df[[f'GLC_{yy}' for yy in range(2013, 2023)]].values; glc_all = np.isin(glc, [10, 11, 12, 20]).all(1)
    res['sample'] = dict(n=n, complete_pct=100 * (nmin == 12).mean(), n8_11_pct=100 * ((nmin >= 8) & (nmin <= 11)).mean(),
                         no_lst_pct=100 * (nmin < 8).mean(), tier2_pct=100 * (tier == 2).mean(), tier1_pct=100 * (tier == 1).mean(),
                         glc_all_pct=100 * glc_all.mean(), grid_cells=ncell,
                         regional_vs_sample_max_abs=float(np.abs(ref_reg - np.nanmedian(A, 0)).max()))

    # ---------------- flags
    I2 = active(A); T2 = signature(I2); Rk2f, _ = trend_flags(A, fixed=0.4545 - 1e-9); R2f = Rk2f.any(1)
    Rk2, _ = trend_flags(A); R2 = Rk2.any(1)
    I3 = active(As); T3 = signature(I3); Rk3, Imp3 = trend_flags(As); R3 = Rk3.any(1)
    assert (mech_code(I2) == df.mechanism.values).all() and (Rk2f == (df[['T_' + k for k in IDX]].fillna(0).values > 0)).all()
    res['v2_published'] = summ(T2, R2f); res['v2_n_aware'] = summ(T2, R2); res['v3'] = summ(T3, R3)
    res['v2_R_lost_n_aware_pct'] = 100 * (R2f & ~R2).sum() / R2f.sum()
    m811 = (nmin >= 8) & (nmin <= 11)
    res['v2_R_in_n8_11'] = [100 * R2f[m811].mean(), 100 * R2[m811].mean()]
    A_reg = A - ref_reg[None]; Ireg = active(A_reg); res['v2_minus_regional'] = summ(signature(Ireg), trend_flags(A_reg)[0].any(1))
    res['v3_detail'] = dict(
        axis_active={k: 100 * I3[:, i].mean() for i, k in enumerate(IDX)}, any_axis=100 * I3.any(1).mean(),
        severity={str(s): 100 * (I3.sum(1) == s).mean() for s in range(6)},
        mech={f'D{d}': 100 * (mech_code(I3) == c).mean() for d, c in ((1, 1), (2, 4), (3, 2), (4, 3))},
        non_exclusive=dict(D1=100 * (I3[:, 0] & I3[:, 2]).mean(), D2=100 * (I3[:, 0] & I3[:, 1] & I3[:, 2]).mean(),
                           D3=100 * (I3[:, 0] & I3[:, 4]).mean(), D4=100 * (I3[:, 0] & I3[:, 3]).mean()),
        worsening={k: 100 * Rk3[:, i].mean() for i, k in enumerate(IDX)}, improving={k: 100 * Imp3[:, i].mean() for i, k in enumerate(IDX)},
        ge2=100 * (Rk3.sum(1) >= 2).mean(), kept_T=100 * (T2 & T3).sum() / T2.sum(), kept_R=100 * (R2 & R3).sum() / R2.sum(),
        T_only=100 * (T3 & ~R3).mean(), R_only=100 * (R3 & ~T3).mean())
    res['v2_detail'] = dict(axis_active={k: 100 * I2[:, i].mean() for i, k in enumerate(IDX)}, any_axis=100 * I2.any(1).mean(),
                            improving={k: 100 * trend_flags(A)[1][:, i].mean() for i, k in enumerate(IDX)},
                            worsening={k: 100 * Rk2[:, i].mean() for i, k in enumerate(IDX)})

    # ---------------- year effect diagnostics on the v2 series
    E2 = np.nan_to_num(A[:, :, 7:], nan=-99) > 1.5
    yc = {}
    for j, yy in enumerate(range(2020, 2025)):
        e = E2.copy(); e[:, :, j] = False
        t = signature(e.sum(-1) >= 3)
        yc[str(yy)] = dict(T_exceed_NDVI=100 * E2[T2, 0, j].mean(), T_without=100 * t.mean(), kept=100 * t.sum() / T2.sum())
    res['v2_year_contribution'] = yc
    full = nmin == 12; per = {}
    for lab, sl, nn in (('2013-2024', slice(0, 12), 12), ('2014-2024', slice(1, 12), 11), ('2013-2023', slice(0, 11), 11)):
        Xs = A[full][:, :, sl]; iu = np.triu_indices(nn, 1); d = Xs[..., iu[1]] - Xs[..., iu[0]]
        tau = np.sign(d).sum(-1) / (nn * (nn - 1) / 2); slope = np.median(d / (iu[1] - iu[0]), -1)
        f = (slope > 0) & (tau >= CRIT[0.05][nn] - 1e-9)
        per[lab] = dict(R=100 * f.any(1).mean(), axes={k: 100 * f[:, i].mean() for i, k in enumerate(IDX)},
                        T_in_R=float((T2[full] & f.any(1)).sum() / T2[full].sum()))
    res['v2_trend_period'] = per
    med = np.nanmedian(A, 0)
    grp2 = {'T only': T2 & ~R2, 'R only': R2 & ~T2, 'T and R': T2 & R2, 'none': ~T2 & ~R2}
    res['v2_group_median_NDVI'] = {g: np.nanmedian(A[m, 0, :], 0).round(3).tolist() for g, m in grp2.items()}

    # ---------------- ERA5-Land covariates
    r = rasterio.open(D + '/era5land_covariates_2013_2026.tif'); E = r.read().astype(float); names = list(r.descriptions)
    col = np.floor((df.lon.values - r.transform.c) / r.transform.a).astype(int); row = np.floor((df.lat.values - r.transform.f) / r.transform.e).astype(int)
    cov = {v: np.stack([E[names.index(f'{v}_{yy}')][row, col] for yy in range(2013, 2027)], 1) for v in ('P', 'PPRE', 'T2M')}
    Z = {v: (X - X[:, :7].mean(1, keepdims=True)) / X[:, :7].std(1, ddof=1, keepdims=True) for v, X in cov.items()}
    res['era5'] = dict(means={v: cov[v].mean(0).round(2).tolist() for v in cov}, zmeans={v: Z[v].mean(0).round(3).tolist() for v in Z},
                       corr={k: {v: float(np.corrcoef(ref_reg[i], Z[v].mean(0)[:12])[0, 1]) for v in Z} for i, k in enumerate(IDX)},
                       baseline_mean={v: float(cov[v][:, :7].mean()) for v in cov}, cells=int(len(np.unique(row * 1000 + col))))
    Xc = np.stack([Z['PPRE'][:, :12], Z['T2M'][:, :12], Z['P'][:, :12]], 2)
    A_cov = {'all': A.copy(), 'base': A.copy()}; cm = {}
    for i, k in enumerate(IDX):
        cm[k] = {}
        for lab, yrs in (('all', list(range(12))), ('base', list(range(7)))):
            yv = A[:, i, :][:, yrs].ravel(); Xv = Xc[:, yrs, :].reshape(-1, 3); ok = ~np.isnan(yv)
            b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), Xv[ok]], yv[ok], rcond=None)[0]
            A_cov[lab][:, i, :] = A[:, i, :] - (b[0] + Xc @ b[1:])
            m0 = np.nanmedian(A[:, i, :], 0); m1 = np.nanmedian(A_cov[lab][:, i, :], 0); ss = lambda v: float(((v - v.mean()) ** 2).sum())
            cm[k][lab] = dict(coef=b[1:].round(3).tolist(), removed_pct=100 * (1 - ss(m1) / ss(m0)))
    res['covariate_model'] = cm
    res['layers_by_adjustment'] = {
        'none': summ(T2, R2), 'covariates_baseline_fit': summ(signature(active(A_cov['base'])), trend_flags(A_cov['base'])[0].any(1)),
        'covariates_all_years_fit': summ(signature(active(A_cov['all'])), trend_flags(A_cov['all'])[0].any(1)),
        'regional_median': res['v2_minus_regional'], 'grid_25km': res['v3']}
    doy = pd.read_csv(D + '/composite_doy_by_year.csv')
    res['doy'] = dict(table=doy.round(2).to_dict('records'), baseline=float(doy[doy.year <= 2019].doy_mean.mean()),
                      evaluation=float(doy[(doy.year >= 2020) & (doy.year <= 2024)].doy_mean.mean()),
                      corr_NDVI=float(np.corrcoef(ref_reg[0], doy.doy_mean[:12])[0, 1]), scenes=int(doy[doy.year <= 2024].scenes.sum()))
    res['spatial_heterogeneity'] = {}
    g = pd.read_csv(D + '/ref_grid_25km_v3_NDVI.csv'); g = g[g['NDVI_2020_count'] >= MIN_PIX]
    res['spatial_heterogeneity']['NDVI_cell_median_p10_p50_p90'] = {str(yy): g[f'NDVI_{yy}_median'].quantile([.1, .5, .9]).round(2).tolist() for yy in YEARS}

    # ---------------- sensitivity: 27 combinations on the v3 series
    S_, tau_, slope_, nv_ = kendall(As); sens = []
    for thr in (1.0, 1.5, 2.0):
        for kk in (2, 3, 4):
            T = signature(active(As, thr, kk))
            for a in (0.10, 0.05, 0.01):
                R = ((slope_ > 0) & (tau_ >= crit_by_n(nv_, a))).any(1); s = summ(T, R)
                sens.append(dict(tau_A=thr, persistence=kk, p=a, crit12=CRIT[a][12], T_km2=s['T_km2'], R_km2=s['R_km2'], TR_km2=s['TR_km2'], J=s['J']))
    res['sensitivity'] = sens
    res['sensitivity_J'] = dict(min=min(s['J'] for s in sens), max=max(s['J'] for s in sens), median=float(np.median([s['J'] for s in sens])))
    p = res['v3']['T'] / 100; res['two_sigma_km2'] = 2 * np.sqrt(p * (1 - p) / n) * AREA
    res['crit_table'] = {str(a): {str(k): round(v, 4) for k, v in CRIT[a].items()} for a in CRIT}

    # ---------------- exchangeable-years scenario (same permutation for the five axes of a pixel)
    alt = {'adopted': lambda F: F.any(1), 'ge2': lambda F: F.sum(1) >= 2, 'ndvi': lambda F: F[:, 0],
           'mirror': lambda F: F[:, 0] & (F[:, 2] | F[:, 4] | F[:, 3])}
    strata = {'all': np.ones(n, bool), 'tier2': tier == 2, 'tier1': tier == 1, 'glc_all': glc_all, 'glc_not': ~glc_all,
              'complete': nmin == 12, 'n8_11': m811}
    rng = np.random.default_rng(20261113)
    acc = {s: {'R': [], 'T': [], 'TR': [], 'J': []} for s in strata}; accalt = {a: [] for a in alt}; peraxis = []
    for _ in range(REPS):
        perm = np.argsort(rng.random((n, 12)), 1); Xp = np.take_along_axis(As, perm[:, None, :], 2)
        F, _ = trend_flags(Xp); Tn = signature(active(restd(Xp))); Rn = F.any(1); peraxis.append(F.mean())
        for s, m in strata.items():
            ss = summ(Tn, Rn, m); [acc[s][k].append(ss[k]) for k in ('R', 'T', 'TR', 'J')]
        for a, fn in alt.items():
            accalt[a].append(100 * fn(F).mean())
    res['scenario'] = {s: {k: float(np.mean(v)) for k, v in d.items()} | {'sd_' + k: float(np.std(v)) for k, v in d.items()} for s, d in acc.items()}
    res['scenario']['per_axis_pct'] = 100 * float(np.mean(peraxis)); res['scenario']['reps'] = REPS
    T3r = signature(active(restd(As)))
    res['observed_restd'] = {s: summ(T3r, R3, m) for s, m in strata.items()}
    res['strata'] = {s: dict(share=100 * m.mean(), v3=summ(T3, R3, m), v2=summ(T2, R2, m)) for s, m in strata.items()}
    res['alt_rules'] = {a: dict(R=100 * fn(Rk3).mean(), scenario=float(np.mean(accalt[a])), **{k: summ(T3, fn(Rk3))[k] for k in ('J', 'R_in_T', 'T_in_R')}) for a, fn in alt.items()}

    # ---------------- serial dependence (complete series)
    Xf = As[full]; tt = np.arange(12.0)
    _, _, b, _ = kendall(Xf); resid = Xf - b[..., None] * tt; resid = resid - resid.mean(-1, keepdims=True)
    def lag1(v): return (v[..., 1:] * v[..., :-1]).sum(-1) / np.maximum((v ** 2).sum(-1), 1e-12)
    r1 = lag1(resid); rngp = np.random.default_rng(7)
    permr = np.take_along_axis(resid, np.argsort(rngp.random(resid.shape), -1), -1); r1p = lag1(permr)
    res['lag1'] = dict(observed={k: float(np.median(r1[:, i])) for i, k in enumerate(IDX)}, white={k: float(np.median(r1p[:, i])) for i, k in enumerate(IDX)})
    # trend-free pre-whitening (Yue et al. 2002): detrend, remove lag-1 dependence, restore trend, test on n-1 values
    Yp = resid[..., 1:] - r1[..., None] * resid[..., :-1] + b[..., None] * tt[1:]
    iu = np.triu_indices(11, 1); d = Yp[..., iu[1]] - Yp[..., iu[0]]
    tau11 = np.sign(d).sum(-1) / 55.0; sl11 = np.median(d / (iu[1] - iu[0]), -1)
    res['tfpw'] = dict(R_before=100 * R3[full].mean(), R_after=100 * ((sl11 > 0) & (tau11 >= CRIT[0.05][11] - 1e-9)).any(1).mean())
    # AR(1) stress scenario: no trend, coefficient 0.3, innovations = permuted detrended residuals of the pixel
    out = []; rngs = np.random.default_rng(11)
    for _ in range(max(20, REPS // 2)):
        perm = np.argsort(rngs.random((len(Xf), 12)), 1); e = np.take_along_axis(resid, perm[:, None, :], 2)
        xs = np.empty_like(e); xs[..., 0] = e[..., 0]
        for j in range(1, 12):
            xs[..., j] = 0.3 * xs[..., j - 1] + e[..., j]
        out.append(100 * trend_flags(xs)[0].any(1).mean())
    res['ar1_stress'] = dict(phi=0.3, R=float(np.mean(out)), sd=float(np.std(out)))

    # ---------------- Benjamini-Hochberg over all pixel-axis tests (exact one-sided p, by n)
    pv = np.ones(S_.shape)
    for nn in range(8, 13):
        N = nn * (nn - 1) // 2; mk = nv_ == nn
        pv[mk] = PGE[nn][np.clip(((N - S_[mk]) / 2).round().astype(int), 0, N)]
    valid = nv_ >= 8; f = np.sort(pv[valid]); m_ = f.size; bh = {}
    for q in (0.05, 0.10, 0.20):
        kq = np.where(f <= q * np.arange(1, m_ + 1) / m_)[0]; thr = f[kq.max()] if kq.size else -1
        keep = valid & (pv <= thr) & (slope_ > 0); bh[str(q)] = dict(pixels=100 * keep.any(1).mean(), tests=100 * keep.sum() / m_)
    res['bh'] = bh; res['bh_tests'] = int(m_)

    # ---------------- spatial dependence and block bootstrap
    tree = cKDTree(np.c_[x, y]); pairs = tree.query_pairs(25000.0, output_type='ndarray')
    dist = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]])
    bins = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000)]; cg = {}
    rngc = np.random.default_rng(3)
    for lab, z in (('T', T3.astype(float)), ('R', R3.astype(float))):
        pz = z.mean(); v = pz * (1 - pz); cg[lab] = {}
        zp = [rngc.permutation(z) for _ in range(99 if not fast else 9)]
        for lo, hi in bins:
            mk = (dist >= lo) & (dist < hi); i_, j_ = pairs[mk, 0], pairs[mk, 1]
            obs = ((z[i_] * z[j_]).mean() - pz ** 2) / v
            refv = [((q[i_] * q[j_]).mean() - pz ** 2) / v for q in zp]
            cg[lab][f'{lo // 1000}-{hi // 1000} km'] = dict(corr=float(obs), ref_max_abs=float(np.max(np.abs(refv))), pairs=int(mk.sum()))
    res['correlogram'] = cg
    boot = {}; rngb = np.random.default_rng(5)
    def block_ranges(size, ox=0.0, oy=0.0, iters=BOOT):
        bid = np.floor((x - ox) / size).astype(int) * 100000 + np.floor((y - oy) / size).astype(int)
        u, inv = np.unique(bid, return_inverse=True); nb = len(u)
        cnt = np.bincount(inv, minlength=nb).astype(float); t = np.bincount(inv, weights=T3, minlength=nb)
        r_ = np.bincount(inv, weights=R3, minlength=nb); tr = np.bincount(inv, weights=(T3 & R3), minlength=nb)
        vals = []
        for _ in range(iters):
            s = rngb.integers(0, nb, nb); c = cnt[s].sum(); a, b_, c2 = t[s].sum(), r_[s].sum(), tr[s].sum()
            vals.append((100 * a / c, 100 * b_ / c, c2 / (a + b_ - c2)))
        q = np.quantile(np.array(vals), [0.025, 0.975], axis=0)
        return dict(blocks=int(nb), T=q[:, 0].round(2).tolist(), R=q[:, 1].round(2).tolist(), J=q[:, 2].round(3).tolist())
    vals = []
    for _ in range(BOOT):
        s = rngb.integers(0, n, n); a, b_, c2 = T3[s].sum(), R3[s].sum(), (T3[s] & R3[s]).sum(); vals.append((100 * a / n, 100 * b_ / n, c2 / (a + b_ - c2)))
    q = np.quantile(np.array(vals), [0.025, 0.975], axis=0)
    boot['pixel'] = dict(T=q[:, 0].round(2).tolist(), R=q[:, 1].round(2).tolist(), J=q[:, 2].round(3).tolist())
    for size in (10000, 25000, 50000):
        boot[f'{size // 1000} km'] = block_ranges(size)
    rngo = np.random.default_rng(9); jlo, jhi = [], []
    for _ in range(10):
        b50 = block_ranges(50000, rngo.uniform(0, 50000), rngo.uniform(0, 50000), iters=max(100, BOOT // 2)); jlo.append(b50['J'][0]); jhi.append(b50['J'][1])
    boot['origin_shift_50km_max_change'] = float(max(np.ptp(jlo), np.ptp(jhi)))
    res['bootstrap'] = boot

    # ---------------- full-raster areas and districts
    a = pd.read_csv(D + '/v3_areas_grid_r2.csv').iloc[0]
    res['raster'] = {k: float(a[k]) for k in a.index if k.endswith('_km2')}
    d = pd.read_csv(D + '/v3_areas_by_district_grid_r2.csv')
    lon = df.groupby('adm2_code').lon.mean(); name2lon = {ADM[c]: lon[c] for c in ADM if c in lon.index}
    d['lon'] = d.ADM2_NAME.map(name2lon); d = d.sort_values('lon')
    for v in ('v2', 'v3'):
        u = d[f'T_{v}_km2'] + d[f'R_{v}_km2'] - d[f'TandR_{v}_km2']
        d['flag_' + v] = u; d['Tonly_' + v] = 100 * (d[f'T_{v}_km2'] - d[f'TandR_{v}_km2']) / u
        d['Ronly_' + v] = 100 * (d[f'R_{v}_km2'] - d[f'TandR_{v}_km2']) / u; d['both_' + v] = 100 * d[f'TandR_{v}_km2'] / u
        d['J_' + v] = d[f'TandR_{v}_km2'] / u; d['Tpct_' + v] = 100 * d[f'T_{v}_km2'] / d.area_km2; d['Rpct_' + v] = 100 * d[f'R_{v}_km2'] / d.area_km2
    res['districts'] = d[['ADM2_NAME', 'lon', 'area_km2', 'flag_v3', 'Tonly_v3', 'Ronly_v3', 'both_v3', 'J_v3', 'Tpct_v3', 'Rpct_v3', 'Tpct_v2', 'Rpct_v2', 'Ronly_v2']].round(3).to_dict('records')
    rngd = np.random.default_rng(1); gr = {}
    for v in ('v2', 'v3'):
        rho = spearmanr(d.lon, d['Ronly_' + v]).statistic
        pp = np.mean([abs(spearmanr(d.lon, rngd.permutation(d['Ronly_' + v].values)).statistic) >= abs(rho) for _ in range(10000)])
        bs = []
        for _ in range(5000):
            s = rngd.integers(0, len(d), len(d))
            if len(np.unique(s)) > 3:
                bs.append(spearmanr(d.lon.values[s], d['Ronly_' + v].values[s]).statistic)
        gr[v] = dict(rho=float(rho), p_perm=float(pp), boot=np.quantile(bs, [0.025, 0.975]).round(2).tolist())
    res['district_gradient'] = gr
    west = d.lon < d.lon.median()
    res['district_west_east'] = {v: dict(west=[float(np.average(d['Tonly_' + v][west], weights=d['flag_' + v][west])), float(np.average(d['Ronly_' + v][west], weights=d['flag_' + v][west]))],
                                         east=[float(np.average(d['Tonly_' + v][~west], weights=d['flag_' + v][~west])), float(np.average(d['Ronly_' + v][~west], weights=d['flag_' + v][~west]))]) for v in ('v2', 'v3')}

    # ---------------- composite index on the v3 series
    def di_of(X, F, thr=1.5, gam=1.0, al=0.6):
        ph = 1 / (1 + np.exp(-gam * (X[:, :, 7:] - thr)))
        dthr = np.nanmean(np.nansum(ph * W[None, :, None], 1), -1); dtr = (F * W).sum(1)
        return al * dthr + (1 - al) * dtr
    DI = di_of(As, Rk3); cls = np.digitize(DI, [0.2, 0.4, 0.6, 0.8])
    cfg = [di_of(As, Rk3, t_, g_, a_) for t_ in (1.0, 1.5, 2.0) for g_ in (1, 2, 3) for a_ in (0.4, 0.6, 0.8)]
    rs = [spearmanr(DI, c).statistic for c in cfg]
    res['DI'] = dict(median=float(np.median(DI)), p95=float(np.quantile(DI, .95)), max=float(DI.max()),
                     classes=[100 * float((cls == i).mean()) for i in range(5)], spearman_min=float(min(rs)), spearman_median=float(np.median(rs)))

    # ---------------- hold-out years
    cell = np.floor(x / CELL).astype(int) * 10000 + np.floor(y / CELL).astype(int)
    def rel(v):
        s = pd.Series(v); gm = s.groupby(cell).transform('median'); gc = s.groupby(cell).transform('count')
        return (s - np.where(gc >= 30, gm, np.nanmedian(v))).values
    Hrel = np.stack([np.stack([rel(H[:, i, j]) for j in range(2)], 1) for i in range(5)], 1)
    def ho(m):
        o = {'n': int(m.sum())}
        for j, yy in enumerate(HOLD):
            v = H[m, 0, j]; w_ = Hrel[m, 0, j]; okv = ~np.isnan(w_)
            wa = Hrel[m][:, :, j]; oka = ~np.isnan(wa).any(1)
            sg = (wa[oka, 0] > 1.5) & ((wa[oka, 2] > 1.5) | (wa[oka, 4] > 1.5) | (wa[oka, 3] > 1.5))
            o[str(yy)] = dict(med=float(np.nanmedian(v)), rel_med=float(np.nanmedian(w_)), rel_exc=100 * float((w_[okv] > 1.5).mean()),
                              exc=100 * float((v[~np.isnan(v)] > 1.5).mean()), signature=100 * float(sg.mean()),
                              axes_exc={k: 100 * float(np.nanmean(Hrel[m, i, j][~np.isnan(Hrel[m, i, j])] > 1.5)) for i, k in enumerate(IDX)})
        return o
    g3 = {'none': ~T3 & ~R3, 'R only': R3 & ~T3, 'T only': T3 & ~R3, 'T and R': T3 & R3}
    g2 = {'none': ~T2 & ~R2, 'R only': R2 & ~T2, 'T only': T2 & ~R2, 'T and R': T2 & R2}
    gk = {'T kept': T2 & T3, 'T dropped': T2 & ~T3, 'R kept': R2 & R3, 'R dropped': R2 & ~R3}
    res['holdout'] = dict(valid_pct={str(yy): 100 * float((~np.isnan(H[:, 0, j])).mean()) for j, yy in enumerate(HOLD)},
                          regional_median={str(yy): {k: float(np.nanmedian(H[:, i, j])) for i, k in enumerate(IDX)} for j, yy in enumerate(HOLD)},
                          v3={g_: ho(m) for g_, m in g3.items()}, v2={g_: ho(m) for g_, m in g2.items()}, kept={g_: ho(m) for g_, m in gk.items()},
                          v3_tier2={g_: ho(m & (tier == 2)) for g_, m in g3.items()}, v3_tier1={g_: ho(m & (tier == 1)) for g_, m in g3.items()})
    # group trajectories of the relative NDVI anomaly, 2013-2026 (in-sample years: grid reference; hold-out years: sample cells)
    res['trajectory_rel_NDVI'] = {g_: np.r_[np.nanmedian(As[m, 0, :], 0), np.nanmedian(Hrel[m, 0, :], 0)].round(3).tolist() for g_, m in g3.items()}

    # ---------------- GEE diagnostics tables
    nvt = pd.read_csv(D + '/diag_n_valid_years_by_axis.csv'); tn = pd.read_csv(D + '/diag_T_R_by_n_min.csv'); sf = pd.read_csv(D + '/sigma_floor_area.csv').iloc[0]
    nd = nvt[nvt.axis == 'NDVI'].set_index('n_valid').km2; ls = nvt[nvt.axis == 'LST'].set_index('n_valid').km2
    res['gee_diag'] = dict(ndvi_n12=100 * nd[12] / nd.sum(), ndvi_n11=100 * nd[11] / nd.sum(), lst_n12=100 * ls[12] / ls.sum(), lst_n0_km2=float(ls[0]), lst_n0_pct=100 * ls[0] / ls.sum(),
                           nmin12_pct=100 * float(tn[tn.n_min == 12].km2.sum() / tn.km2.sum()), nmin7_11_pct=100 * float(tn[(tn.n_min >= 7) & (tn.n_min <= 11)].km2.sum() / tn.km2.sum()),
                           sigma_floor={k: 100 * sf[k + '_floor_km2'] / sf['area_km2'] for k in IDX})
    res['regional_median_full_raster'] = {k: ref_reg[i].round(3).tolist() for i, k in enumerate(IDX)}
    supplement(res, df, A, H, x, y, ref_reg, ref)
    json.dump(res, open(O + '/results.json', 'w', encoding='utf8'), indent=1, default=float)
    print('done in %.0f s' % (time.time() - t0))

if __name__ == '__main__':
    main()
