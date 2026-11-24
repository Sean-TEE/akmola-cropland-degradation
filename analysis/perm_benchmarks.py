#!/usr/bin/env python3
"""MITT-CDA: permutation benchmarks of the whole procedure (Section 2.3).
Usage: python perm_benchmarks.py DATA_DIR OUT_DIR MODE [REPS_WHOLE] [REPS_PIXEL]     MODE: pixel | shrunk
For every repetition the years of the indicator values and of the precipitation are permuted together, and the procedure of
procedure.run() re-estimates the baseline median, the scale (with its shrinkage weights), the standardized precipitation and the
25 km reference before the decision rules are applied. No re-standardization is involved; the observed order goes through the same
procedure.
  whole_12  one permutation of 2013-2024 for all pixels (layer sizes, overlap, direction of trends)
  whole_14  one permutation of 2013-2026 for all pixels; flags from the first twelve positions, exceedance in the last two
  pixel_12  an independent permutation of 2013-2024 for every pixel (layer sizes; trend layer by stratum)"""
import sys, json, time
import numpy as np, pandas as pd
import procedure as PR, mittcda_v3_analysis as M

D, O, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
RW = int(sys.argv[4]) if len(sys.argv) > 4 else 200; RP = int(sys.argv[5]) if len(sys.argv) > 5 else 40
t0 = time.time(); d = PR.load(D); X, P, C, df = d['X'], d['P'], d['C'], d['df']; n = len(df)
rng = np.random.default_rng({'pixel': 711, 'shrunk': 712, 'pooled': 713}[MODE])
tier = df.tier.values; glc_all = np.isin(df[[f'GLC_{yy}' for yy in range(2013, 2023)]].values, [10, 11, 12, 20]).all(1)
nmin = (~np.isnan(d['A'][:, :, :12])).sum(2).min(1)
STRATA = {'all': np.ones(n, bool), 'tier2': tier == 2, 'tier1': tier == 1, 'glc_all': glc_all, 'glc_not': ~glc_all,
          'complete': nmin == 12, 'n8_11': (nmin >= 8) & (nmin <= 11), 'stable': (tier == 2) & glc_all}

def sizes(V, incl):
    I, T, Rk, Imp = PR.flags(V, incl); R = Rk.any(1); m = incl
    t, r, i = T[m].mean(), R[m].mean(), (T & R)[m].mean()
    return dict(T=100 * t, R=100 * r, TR=100 * i, J=i / (t + r - i) if t + r - i > 0 else np.nan, improving=100 * Imp.any(1)[m].mean(),
                diff=100 * (R[m].mean() - Imp.any(1)[m].mean())), T, R, Rk

def holdout(V, incl, T, R, Rk):
    out = {}
    G = {'T': T, 'T and R': T & R, 'T only': T & ~R, 'R only': R & ~T, 'none': ~T & ~R & incl}
    for j in (0, 1):
        v = V[:, 0, 12 + j]; ok = ~np.isnan(v) & incl; ex = np.nan_to_num(v, nan=-99) > 1.5
        for g, m in G.items():
            mm = m & ok; out['%s|%d' % (g, j)] = 100 * ex[mm].mean() if mm.any() else np.nan
        H = np.nan_to_num(V[:, :, 12 + j], nan=-99) > 1.5; okk = ~np.isnan(V[:, :, 12 + j])
        own = (H & Rk & okk).any(1); val = (Rk & okk).any(1); m = G['R only'] & val & incl
        out['R only own axes|%d' % j] = 100 * own[m].mean() if m.any() else np.nan
        for g in ('T', 'T and R', 'T only', 'R only', 'R only own axes'):
            out['%s-none|%d' % (g, j)] = out['%s|%d' % (g, j)] - out['none|%d' % j]
        out['TandR-Tonly|%d' % j] = out['T and R|%d' % j] - out['T only|%d' % j]
    return out

def observed():
    r = PR.run(X[:, :, :12], P[:, :12], C, MODE); s, T, R, Rk = sizes(r['V'], r['incl'])
    r14 = PR.run(X, P, C, MODE); s14, T14, R14, Rk14 = sizes(r14['V'], r14['incl'])
    assert (T == T14).all() and (R == R14).all()                  # hold-out years do not enter the baseline or the flags
    st = {k: 100 * float(R[m].mean()) for k, m in STRATA.items()}
    return s, holdout(r14['V'], r14['incl'], T14, R14, Rk14), st

res = {'mode': MODE, 'n': n}
obs_s, obs_h, obs_st = observed(); res['observed'] = dict(sizes=obs_s, holdout=obs_h, R_by_stratum=obs_st)
print(MODE, 'observed', json.dumps(obs_s), '%.0f s' % (time.time() - t0), flush=True)
def summarize(rows, obs):
    d_ = pd.DataFrame(rows); out = {}
    for k in d_.columns:
        v = d_[k].dropna().values
        p = float((1 + (v >= obs[k]).sum()) / (len(v) + 1)) if k in obs and obs[k] == obs[k] else None
        out[k] = dict(mean=float(v.mean()), sd=float(v.std()), q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975)), p=p)
        if k == 'diff':
            out[k]['p_two_sided'] = float((1 + (np.abs(v) >= abs(obs[k])).sum()) / (len(v) + 1))
    return out
rows = []
for _ in range(RW):
    p = np.argsort(rng.random(12)); r = PR.run(X[:, :, p], P[:, p], C, MODE); rows.append(sizes(r['V'], r['incl'])[0])
res['whole_12'] = summarize(rows, obs_s); res['whole_12']['reps'] = RW
print(MODE, 'whole_12 %.0f s' % (time.time() - t0), flush=True); json.dump(res, open(O + '/perm_%s.json' % MODE, 'w'), indent=1, default=float)
rows = []
for _ in range(RW):
    p = np.argsort(rng.random(14)); r = PR.run(X[:, :, p], P[:, p], C, MODE)
    s, T, R, Rk = sizes(r['V'], r['incl']); rows.append(holdout(r['V'], r['incl'], T, R, Rk))
res['whole_14'] = summarize(rows, obs_h); res['whole_14']['reps'] = RW
print(MODE, 'whole_14 %.0f s' % (time.time() - t0), flush=True); json.dump(res, open(O + '/perm_%s.json' % MODE, 'w'), indent=1, default=float)
rows = []; strat = {k: [] for k in STRATA}
for _ in range(RP):
    p = np.argsort(rng.random((n, 12)), 1); Xp = np.take_along_axis(X[:, :, :12], p[:, None, :], 2); Pp = np.take_along_axis(P[:, :12], p, 1)
    r = PR.run(Xp, Pp, C, MODE); s, T, R, Rk = sizes(r['V'], r['incl']); rows.append(s)
    for k, m in STRATA.items(): strat[k].append(100 * float(R[m & r['incl']].mean()))
res['pixel_12'] = summarize(rows, obs_s); res['pixel_12']['reps'] = RP; res['pixel_12_R_by_stratum'] = {k: float(np.mean(v)) for k, v in strat.items()}
json.dump(res, open(O + '/perm_%s.json' % MODE, 'w'), indent=1, default=float)
print(MODE, 'done %.0f s' % (time.time() - t0), flush=True)
