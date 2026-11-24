#!/usr/bin/env python3
"""MITT-CDA: all descriptive numbers of the manuscript from one procedure (procedure.run) applied to the observed order of years.
Usage: python compute_numbers.py DATA_DIR OUT_DIR       (the permutation benchmarks come from perm_benchmarks.py)"""
import sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import procedure as PR, mittcda_v3_analysis as M, stepb as S

D, O = sys.argv[1], sys.argv[2]; t0 = time.time(); rng = np.random.default_rng(20261120)
d = PR.load(D); X, P, C, df, A = d['X'], d['P'], d['C'], d['df'], d['A']; n = len(df); x, y = d['x'], d['y']; AREA = M.AREA; W = M.W
RUN = {m: PR.run(X, P, C, m) for m in ('pixel', 'shrunk', 'pooled')}
SER = {'uncorrected': (A, np.ones(n, bool)), 'pixel_scale': (RUN['pixel']['V'], RUN['pixel']['incl']), 'shrunk_scale': (RUN['shrunk']['V'], RUN['shrunk']['incl']),
       'pooled_scale': (RUN['pooled']['V'], RUN['pooled']['incl'])}
tier = df.tier.values; glc_all = np.isin(df[[f'GLC_{yy}' for yy in range(2013, 2023)]].values, [10, 11, 12, 20]).all(1)
nmin = (~np.isnan(A[:, :, :12])).sum(2).min(1); stable = (tier == 2) & glc_all
b50 = np.floor(x / 50000).astype(int) * 1000 + np.floor(y / 50000).astype(int); bid, binv = np.unique(b50, return_inverse=True); nb50 = len(bid)
c125 = np.floor(x / 12500).astype(int) * 10000 + np.floor(y / 12500).astype(int)

def boot_share(masks, iters=1000):
    cnt = np.bincount(binv, minlength=nb50).astype(float); sums = {k: np.bincount(binv, weights=m.astype(float), minlength=nb50) for k, m in masks.items()}
    out = {k: [] for k in masks}; out['J'] = []
    for _ in range(iters):
        w = np.bincount(rng.integers(0, nb50, nb50), minlength=nb50).astype(float); N = (w * cnt).sum(); v = {k: (w * s).sum() / N for k, s in sums.items()}
        for k in masks: out[k].append(100 * v[k])
        out['J'].append(v['TR'] / (v['T'] + v['R'] - v['TR']))
    return {k: np.quantile(vv, [.025, .975]).round(3).tolist() for k, vv in out.items()}

def boot_ratio(num, den, iters=1000):
    a = np.bincount(binv, weights=num.astype(float), minlength=nb50); b = np.bincount(binv, weights=den.astype(float), minlength=nb50); vals = []
    for _ in range(iters):
        w = np.bincount(rng.integers(0, nb50, nb50), minlength=nb50); vals.append(100 * (w * a).sum() / max((w * b).sum(), 1))
    return np.quantile(vals, [.025, .975]).round(1).tolist()

res = {'n': n, 'area_km2': AREA, 'blocks50': int(nb50)}; FL = {}
for lab, (V, incl) in SER.items():
    I, T, Rk, Imp = PR.flags(V, incl); R = Rk.any(1); mc = M.mech_code(I); FL[lab] = (I, T, Rk, Imp, R, mc)
    s = M.summ(T, R); p = lambda m: 100 * float(m.mean()); km = lambda m: float(m.mean() * AREA)
    two = lambda m: float(2 * np.sqrt(m.mean() * (1 - m.mean()) / n) * AREA)
    E = np.nan_to_num(V[:, :, 7:12], nan=-99) > 1.5; jt = lambda m: m.sum(-1) >= 3
    nd, nm, ls, tg, si = (E[:, i] for i in range(5))
    L = dict(T=s['T'], R=s['R'], TR=s['TR'], U=s['T'] + s['R'] - s['TR'], J=s['J'], R_in_T=s['R_in_T'], T_in_R=s['T_in_R'],
             T_km2=km(T), R_km2=km(R), TR_km2=km(T & R), U_km2=km(T | R), Tonly_km2=km(T & ~R), Ronly_km2=km(R & ~T),
             two_sigma_km2={'T': two(T), 'R': two(R), 'TR': two(T & R)}, severity=[p(I.sum(1) == i) for i in range(6)], any_axis=p(I.any(1)),
             mech={f'D{dd}': dict(pct=p(mc == c), km2=km(mc == c)) for dd, c in ((1, 1), (2, 4), (3, 2), (4, 3))},
             mech_nonexclusive={'D1': p(I[:, 0] & I[:, 2]), 'D2': p(I[:, 0] & I[:, 1] & I[:, 2]), 'D3': p(I[:, 0] & I[:, 4]), 'D4': p(I[:, 0] & I[:, 3])},
             T_joint_seasons=p((jt(nd & ls) | jt(nd & si) | jt(nd & tg)) & incl), T_joint_in_T=100 * float(((jt(nd & ls) | jt(nd & si) | jt(nd & tg)) & T).sum() / max(T.sum(), 1)),
             worsening_axis={k: dict(pct=p(Rk[:, i]), km2=km(Rk[:, i])) for i, k in enumerate(M.IDX)}, ge2_worsening=dict(pct=p(Rk.sum(1) >= 2), km2=km(Rk.sum(1) >= 2)),
             improving_axis={k: p(Imp[:, i]) for i, k in enumerate(M.IDX)}, improving_any=p(Imp.any(1)),
             exceed_NDVI_by_year=[round(100 * float(np.mean(V[:, 0, j][~np.isnan(V[:, 0, j])] > 1.5)), 1) for j in range(14)],
             spread_NDVI_by_year=[round(float(1.4826 * np.nanmedian(np.abs(V[:, 0, j] - np.nanmedian(V[:, 0, j])))), 2) for j in range(14)])
    L['boot50'] = boot_share({'T': T, 'R': R, 'TR': T & R}); res[lab] = L
    print(lab, 'T %.3f R %.3f' % (L['T'], L['R']), flush=True)
I2, T2, Rk2, Imp2, R2, mc2 = FL['shrunk_scale']; V2, inc2 = SER['shrunk_scale']; I0, T0, Rk0, Imp0, R0, mc0 = FL['pixel_scale']
# ---- diagnostics of the scale: ratio to the cell, slope on the cell median, overlap of the two scales
ratio = RUN['pixel']['raw'][:, 0] / RUN['pixel']['sc'][:, 0]
def slope_on_cell(Aq, m):
    Y = Aq[:, 0, :12]; Xr = np.stack([C.median(Y[:, j][:, None])[:, 0] for j in range(12)], 1); ok = ~np.isnan(Y); Xc = np.where(ok, Xr, np.nan)
    g = np.nansum((Xc - np.nanmean(Xc, 1, keepdims=True)) * (Y - np.nanmean(Y, 1, keepdims=True)), 1) / np.nansum((Xc - np.nanmean(Xc, 1, keepdims=True)) ** 2, 1)
    return float(np.nanmedian(g[m]))
res['diag'] = dict(ratio_NDVI={'T pixel': float(np.median(ratio[T0])), 'none pixel': float(np.median(ratio[~T0 & ~R0])), 'R only pixel': float(np.median(ratio[R0 & ~T0])),
                               'T shrunk': float(np.median(ratio[T2])), 'none shrunk': float(np.median(ratio[~T2 & ~R2]))},
                   ratio_T_pixel_axes={k: float(np.nanmedian((RUN['pixel']['raw'][:, i] / RUN['pixel']['sc'][:, i])[T0])) for i, k in enumerate(M.IDX)},
                   slope={'T pixel': slope_on_cell(RUN['pixel']['Aq'], T0), 'none pixel': slope_on_cell(RUN['pixel']['Aq'], ~T0 & ~R0),
                          'T shrunk': slope_on_cell(RUN['shrunk']['Aq'], T2), 'none shrunk': slope_on_cell(RUN['shrunk']['Aq'], ~T2 & ~R2)},
                   overlap=dict(T_pixel=int(T0.sum()), T_shrunk=int(T2.sum()), T_both=int((T0 & T2).sum()), R_pixel=int(R0.sum()), R_both=int((R0 & R2).sum()),
                                ratio_left=float(np.median(ratio[T0 & ~T2])), ratio_new=float(np.median(ratio[T2 & ~T0]))),
                   weights={k: float(np.nanmax(RUN['shrunk']['w'][:, i])) for i, k in enumerate(M.IDX)},
                   v_obs={k: float(v) for k, v in zip(M.IDX, d['C'].within_var(np.log(np.where(RUN['shrunk']['raw'] > 0, RUN['shrunk']['raw'], np.nan))))},
                   zero_mad={k: int((RUN['shrunk']['raw'][:, i] == 0).sum()) for i, k in enumerate(M.IDX)},
                   floor_binds_shrunk={k: 100 * float((RUN['shrunk']['st'][:, i] <= S.FLOOR[i] + 1e-12).mean()) for i, k in enumerate(M.IDX)})
ex = lambda m, j: (np.nan_to_num(V2[:, 0, 12 + j], nan=-99) > 1.5)[m & ~np.isnan(V2[:, 0, 12 + j])]
res['diag']['holdout_by_overlap'] = {g: [round(100 * float(ex(m, j).mean()), 1) for j in (0, 1)] for g, m in
                                     (('both', T0 & T2), ('left', T0 & ~T2), ('new', T2 & ~T0), ('neither', ~T0 & ~T2 & inc2))}
# ---- uncorrected: share of the uncorrected layers that disappears with local normalization (pixel scale)
Iu, Tu, Rku, Impu, Ru, mcu = FL['uncorrected']
res['uncorrected_lost'] = dict(T_lost_pct=100 * float((Tu & ~T0).sum() / Tu.sum()), R_lost_pct=100 * float((Ru & ~R0).sum() / Ru.sum()),
                               T_new=int((T0 & ~Tu).sum()), R_new=int((R0 & ~Ru).sum()))
# ---- sensitivity, ties, trend tests, slopes
S_, tau_, slope_, nv_ = M.kendall(V2[:, :, :12]); sens = []
for thr in (1.0, 1.5, 2.0):
    for kk in (2, 3, 4):
        T = M.signature(M.active(V2[:, :, :12], thr, kk)) & inc2
        for a in (0.10, 0.05, 0.01):
            R = ((slope_ > 0) & (tau_ >= M.crit_by_n(nv_, a))).any(1); s = M.summ(T, R)
            sens.append(dict(tau_A=thr, persistence=kk, p=a, crit12=M.CRIT[a][12], T_km2=s['T_km2'], R_km2=s['R_km2'], TR_km2=s['TR_km2'], J=s['J'], T_in_adopted=100 * float((T & T2).sum() / max(T2.sum(), 1))))
res['sensitivity'] = sens
ser = 0; tie = 0
for i in range(5):
    for r_ in V2[:, i, :12]:
        v = r_[~np.isnan(r_)]
        if len(v) >= 8: ser += 1; tie += int(len(np.unique(v)) < len(v))
res['ties'] = dict(series=ser, with_ties=tie)
pp = np.ones_like(tau_); ok = nv_ >= 8
for nn in range(8, 13):
    m = ok & (nv_ == nn); inv = np.clip(np.floor((nn * (nn - 1) / 2 - S_[m]) / 2 + 1e-9).astype(int), 0, len(M.PGE[nn]) - 1); pp[m] = M.PGE[nn][inv]
o = np.sort(pp[ok]); mt = len(o); bh = {}
for q in (0.05, 0.10, 0.20):
    pas = o <= q * np.arange(1, mt + 1) / mt; k = int(np.nonzero(pas)[0].max() + 1) if pas.any() else 0; thr = o[k - 1] if k else -1
    bh[str(q)] = dict(tests=k, pixels_pct=100 * float((ok & (pp <= thr) & (slope_ > 0)).any(1).mean()))
res['bh'] = bh; res['bh_tests'] = int(mt)
res['slope_flagged_sigma_per_decade'] = {k: np.quantile(10 * slope_[Rk2[:, i], i], [.1, .5, .9]).round(2).tolist() for i, k in enumerate(M.IDX)}
tt = np.arange(12.0)
def lag1_med(Xs):
    out = []
    for i in range(5):
        Xi = Xs[:, i, :]; Xi = Xi[~np.isnan(Xi).any(1)]; dd = Xi[:, None, :] - Xi[:, :, None]; iu = np.triu_indices(12, 1)
        sl = np.median(dd[:, iu[0], iu[1]] / (tt[None, :] - tt[:, None])[iu], 1); r = Xi - sl[:, None] * tt; r = r - r.mean(1, keepdims=True)
        out.append(float(np.median((r[:, 1:] * r[:, :-1]).sum(1) / np.maximum((r ** 2).sum(1), 1e-12))))
    return out
res['lag1'] = dict(observed=lag1_med(V2[:, :, :12]), random=lag1_med(np.take_along_axis(V2[:, :, :12], np.argsort(rng.random((n, 12)), 1)[:, None, :], 2)))
# ---- strata
strata = {'all': inc2, 'tier2': inc2 & (tier == 2), 'tier1': inc2 & (tier == 1), 'glc_all': inc2 & glc_all, 'glc_not': inc2 & ~glc_all,
          'complete': inc2 & (nmin == 12), 'n8_11': inc2 & (nmin >= 8) & (nmin <= 11), 'stable': inc2 & stable}
res['strata'] = {k: dict(share=100 * float(m.mean()), **{kk: v for kk, v in M.summ(T2, R2, m).items() if kk in ('T', 'R', 'TR', 'J', 'R_in_T', 'T_in_R')}) for k, m in strata.items()}
# ---- composite index (a missing axis contributes zero to both scores)
def di_of(Xs, F, thr=1.5, gam=1.0, al=0.6):
    ph = 1 / (1 + np.exp(-gam * (Xs[:, :, 7:12] - thr))); dthr = np.nanmean(np.nansum(ph * W[None, :, None], 1), -1); dtr = (F * W).sum(1)
    return al * dthr + (1 - al) * dtr
DI = di_of(V2, Rk2); cls = np.digitize(DI, [0.2, 0.4, 0.6, 0.8]); cfg = [di_of(V2, Rk2, a, g, al) for a in (1.0, 1.5, 2.0) for g in (1, 2, 3) for al in (0.4, 0.6, 0.8)]
top = DI >= np.quantile(DI, .99)
res['DI'] = dict(median=float(np.median(DI)), p95=float(np.quantile(DI, .95)), max=float(DI.max()), classes=[100 * float((cls == i).mean()) for i in range(5)],
                 spearman_min=float(min(spearmanr(DI, c).statistic for c in cfg)), spearman_median=float(np.median([spearmanr(DI, c).statistic for c in cfg])),
                 top1_overlap_min=100 * float(min((top & (c >= np.quantile(c, .99))).sum() / top.sum() for c in cfg)))
# ---- hold-out groups (adopted procedure)
G = {'T and R': T2 & R2, 'T only': T2 & ~R2, 'R only': R2 & ~T2, 'none': ~T2 & ~R2 & inc2, 'T': T2, 'R': R2}
ho = {}
for g, m in G.items():
    o_ = {'n': int(m.sum())}
    for j in (0, 1):
        v = V2[:, 0, 12 + j]; okv = m & ~np.isnan(v); e = (np.nan_to_num(v, nan=-99) > 1.5) & okv
        wa = V2[:, :, 12 + j]; oka = m & ~np.isnan(wa).any(1); sg = (wa[oka, 0] > 1.5) & ((wa[oka, 2] > 1.5) | (wa[oka, 4] > 1.5) | (wa[oka, 3] > 1.5))
        o_[str(2025 + j)] = dict(valid=int(okv.sum()), exceed_n=int(e.sum()), exceed=100 * float(e.sum() / max(okv.sum(), 1)), exceed_boot=boot_ratio(e, okv),
                                 median=float(np.nanmedian(v[okv])), signature=100 * float(sg.mean()),
                                 tier2=100 * float(e[tier == 2].sum() / max((okv & (tier == 2)).sum(), 1)), tier1=100 * float(e[tier == 1].sum() / max((okv & (tier == 1)).sum(), 1)),
                                 stable=100 * float(e[stable].sum() / max((okv & stable).sum(), 1)))
    o_['traj'] = np.nanmedian(V2[m, 0, :], 0).round(2).tolist(); ho[g] = o_
for j in (0, 1):   # trend-only pixels on the axes of their trend
    H = np.nan_to_num(V2[:, :, 12 + j], nan=-99) > 1.5; okk = ~np.isnan(V2[:, :, 12 + j]); own = (H & Rk2 & okk).any(1); val = G['R only'] & (Rk2 & okk).any(1)
    ho['R only'][str(2025 + j)]['own_axes'] = 100 * float(own[val].mean()); ho['R only'][str(2025 + j)]['own_axes_boot'] = boot_ratio(own & val, val)
    ho['R only'][str(2025 + j)]['own_axes_valid'] = int(val.sum()); ho['R only'][str(2025 + j)]['own_axes_n'] = int((own & val).sum())
for j in (0, 1):   # direct difference T and R minus T only, 50 km block bootstrap
    v = V2[:, 0, 12 + j]; okv = ~np.isnan(v); e = np.nan_to_num(v, nan=-99) > 1.5
    a1, b1 = G['T and R'] & okv, G['T only'] & okv
    A1 = np.bincount(binv, weights=(e & a1).astype(float), minlength=nb50); N1 = np.bincount(binv, weights=a1.astype(float), minlength=nb50)
    A2_ = np.bincount(binv, weights=(e & b1).astype(float), minlength=nb50); N2 = np.bincount(binv, weights=b1.astype(float), minlength=nb50); vals = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, nb50, nb50), minlength=nb50); vals.append(100 * ((w * A1).sum() / max((w * N1).sum(), 1) - (w * A2_).sum() / max((w * N2).sum(), 1)))
    ho['TandR_minus_Tonly_%d' % (2025 + j)] = dict(diff=100 * float(e[a1].mean() - e[b1].mean()), boot=np.quantile(vals, [.025, .975]).round(1).tolist(), share_le0=float(np.mean(np.array(vals) <= 0)))
for j in (0, 1):   # threshold only minus trend only, both on NDVI, 50 km block bootstrap (after all earlier random draws)
    v = V2[:, 0, 12 + j]; okv = ~np.isnan(v); e = np.nan_to_num(v, nan=-99) > 1.5
    a1, b1 = G['T only'] & okv, G['R only'] & okv
    A1 = np.bincount(binv, weights=(e & a1).astype(float), minlength=nb50); N1 = np.bincount(binv, weights=a1.astype(float), minlength=nb50)
    A2_ = np.bincount(binv, weights=(e & b1).astype(float), minlength=nb50); N2 = np.bincount(binv, weights=b1.astype(float), minlength=nb50); vals = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, nb50, nb50), minlength=nb50); vals.append(100 * ((w * A1).sum() / max((w * N1).sum(), 1) - (w * A2_).sum() / max((w * N2).sum(), 1)))
    ho['Tonly_minus_Ronly_%d' % (2025 + j)] = dict(diff=100 * float(e[a1].mean() - e[b1].mean()), boot=np.quantile(vals, [.025, .975]).round(1).tolist())
res['holdout'] = ho; res['holdout_coverage'] = {str(2025 + j): 100 * float((~np.isnan(V2[:, 0, 12 + j]) & inc2).mean()) for j in (0, 1)}
# ---- maps and scale ratio
cu, cinv = np.unique(c125, return_inverse=True); cc = np.bincount(cinv)
mp = {'cx': ((cu // 10000) * 12500 + 6250).tolist(), 'cy': ((cu % 10000) * 12500 + 6250).tolist(), 'count': cc.tolist()}
for lab in ('uncorrected', 'pixel_scale', 'shrunk_scale'):
    _, T, _, _, R, _ = FL[lab]; mp['T_' + lab] = (np.bincount(cinv, weights=T.astype(float)) / cc).round(4).tolist(); mp['R_' + lab] = (np.bincount(cinv, weights=R.astype(float)) / cc).round(4).tolist()
json.dump(mp, open(O + '/maps.json', 'w'))
np.savez_compressed(O + '/scale_ratio.npz', T0=ratio[T0], N0=ratio[~T0 & ~R0], T2=ratio[T2], N2=ratio[~T2 & ~R2])
res['lat_range'] = [float(df.lat.min()), float(df.lat.max())]
res['R_only_not_NDVI_pct'] = 100 * float((~Rk2[R2 & ~T2, 0]).mean())
json.dump(res, open(O + '/numbers.json', 'w'), indent=1, default=float); print('done %.0f s' % (time.time() - t0))
