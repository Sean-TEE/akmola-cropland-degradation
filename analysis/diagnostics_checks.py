#!/usr/bin/env python3
"""Supplementary checks: serial dependence of the final (shrunk-scale) relative anomalies, shrinkage weights by
number of baseline values, and PCA weight definition. Usage: python diagnostics_checks.py DATA_DIR OUT_JSON"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else '.')
import procedure as PR, mittcda_v3_analysis as M, stepb as S
D, O = sys.argv[1], sys.argv[2]
d = PR.load(D); r = PR.run(d['X'], d['P'], d['C'], 'shrunk'); V, incl = r['V'], r['incl']
res = {}
# shrinkage weights by number of baseline values
lnr = np.where(r['raw'] > 0, np.log(np.where(r['raw'] > 0, r['raw'], 1.0)), np.nan); vobs = d['C'].within_var(lnr)
nb = (~np.isnan(d['X'][:, :, :7])).sum(-1)
res['v_obs'] = dict(zip(M.IDX, vobs.round(4).tolist()))
res['omega_by_n'] = {n: dict(zip(M.IDX, np.clip(1 - v / vobs, 0, 1).round(3).tolist())) for n, v in S.V_EST.items()}
res['share_by_n'] = {n: dict(zip(M.IDX, (100 * ((nb == n) & incl[:, None]).sum(0) / incl.sum()).round(2).tolist())) for n in (5, 6, 7)}
# serial dependence, complete series, shrunk scale
tt = np.arange(12.0); X12 = V[:, :, :12]
out = {}; Rb = []; Ra = []; full_any = None
rng = np.random.default_rng(20261124)
for i, k in enumerate(M.IDX):
    Xi = X12[:, i, :]; ok = incl & ~np.isnan(Xi).any(1); Xi = Xi[ok]
    iu = np.triu_indices(12, 1); dd = Xi[:, iu[1]] - Xi[:, iu[0]]; b = np.median(dd / (iu[1] - iu[0]), 1)
    e = Xi - b[:, None] * tt; e = e - e.mean(1, keepdims=True)
    lag = lambda v: (v[:, 1:] * v[:, :-1]).sum(1) / np.maximum((v ** 2).sum(1), 1e-12)
    r1 = lag(e)
    Xr = np.take_along_axis(Xi, np.argsort(rng.random(Xi.shape), 1), 1)                 # series in random order, detrended again
    ddr = Xr[:, iu[1]] - Xr[:, iu[0]]; br = np.median(ddr / (iu[1] - iu[0]), 1); er = Xr - br[:, None] * tt; er = er - er.mean(1, keepdims=True)
    r1r = lag(er)
    q = lambda a: dict(median=round(float(np.median(a)), 3), p75=round(float(np.quantile(a, .75)), 3), p90=round(float(np.quantile(a, .9)), 3),
                       pos=round(100 * float((a > 0).mean()), 1), gt03=round(100 * float((a > 0.3).mean()), 1))
    out[k] = dict(n=int(ok.sum()), observed=q(r1), random=q(r1r))
    # TFPW (Yue et al. 2002) on n-1 values
    Yp = e[:, 1:] - r1[:, None] * e[:, :-1] + b[:, None] * tt[1:]
    j = np.triu_indices(11, 1); d2 = Yp[:, j[1]] - Yp[:, j[0]]
    tau11 = np.sign(d2).sum(1) / 55.0; sl11 = np.median(d2 / (j[1] - j[0]), 1)
    Sx, tau, sl, nv = M.kendall(Xi[:, None, :])
    before = (sl[:, 0] > 0) & (tau[:, 0] >= M.CRIT[0.05][12] - 1e-9); after = (sl11 > 0) & (tau11 >= M.CRIT[0.05][11] - 1e-9)
    out[k].update(R_before=round(100 * float(before.mean()), 2), R_after_tfpw=round(100 * float(after.mean()), 2))
res['lag1_shrunk_complete'] = out
# R (any axis) for pixels complete on all five axes
okall = incl & ~np.isnan(X12).any((1, 2)); Xa = X12[okall]
Rk, _ = M.trend_flags(Xa); res['R_complete_all_axes'] = dict(n=int(okall.sum()), R=round(100 * float(Rk.any(1).mean()), 2))
_, _, ba, _ = M.kendall(Xa); ea = Xa - ba[..., None] * tt; ea = ea - ea.mean(-1, keepdims=True)
r1a = (ea[..., 1:] * ea[..., :-1]).sum(-1) / np.maximum((ea ** 2).sum(-1), 1e-12)
Yp = ea[..., 1:] - r1a[..., None] * ea[..., :-1] + ba[..., None] * tt[1:]; j = np.triu_indices(11, 1); d2 = Yp[..., j[1]] - Yp[..., j[0]]
tau11 = np.sign(d2).sum(-1) / 55.0; sl11 = np.median(d2 / (j[1] - j[0]), -1)
res['R_complete_all_axes']['R_tfpw'] = round(100 * float(((sl11 > 0) & (tau11 >= M.CRIT[0.05][11] - 1e-9)).any(1).mean()), 2)
outs = []; rs = np.random.default_rng(11)
for _ in range(20):
    perm = np.argsort(rs.random((len(Xa), 12)), 1); e = np.take_along_axis(ea, perm[:, None, :], 2)
    xs = np.empty_like(e); xs[..., 0] = e[..., 0]
    for t in range(1, 12): xs[..., t] = 0.3 * xs[..., t - 1] + e[..., t]
    outs.append(100 * M.trend_flags(xs)[0].any(1).mean())
res['R_complete_all_axes']['ar1_stress_R'] = round(float(np.mean(outs)), 2)
# PCA weights from the correlation matrix of Table 3 (two decimals)
Cm = np.array([[1, .72, .31, .37, .63], [.72, 1, .31, .12, .33], [.31, .31, 1, .16, .27], [.37, .12, .16, 1, .36], [.63, .33, .27, .36, 1]])
lam, vec = np.linalg.eigh(Cm); o = np.argsort(lam)[::-1]; lam, vec = lam[o], vec[:, o]
ev = lam / lam.sum(); res['pca_explained'] = (100 * ev[:3]).round(1).tolist()
wa = (np.abs(vec[:, :3]) * ev[:3]).sum(1); wb = (np.abs(vec[:, :3] * np.sqrt(lam[:3])) * ev[:3]).sum(1)
res['pca_w_eigvec'] = (wa / wa.sum()).round(3).tolist(); res['pca_w_loading'] = (wb / wb.sum()).round(3).tolist()
res['pca_w_paper'] = [0.200, 0.218, 0.212, 0.194, 0.176]
json.dump(res, open(O, 'w'), indent=1); print(json.dumps(res, indent=1))
