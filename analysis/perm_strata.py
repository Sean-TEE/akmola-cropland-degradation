#!/usr/bin/env python3
"""MITT-CDA: trend layer by stratum under the whole-area permutation of the whole procedure (shrunk scale).
Usage: python perm_strata.py DATA_DIR OUT_DIR [REPS]"""
import sys, json, time
import numpy as np
import procedure as PR
D, O = sys.argv[1], sys.argv[2]; REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
d = PR.load(D); X, P, C, df = d['X'], d['P'], d['C'], d['df']; n = len(df); rng = np.random.default_rng(714); t0 = time.time()
tier = df.tier.values; glc_all = np.isin(df[[f'GLC_{yy}' for yy in range(2013, 2023)]].values, [10, 11, 12, 20]).all(1)
nmin = (~np.isnan(d['A'][:, :, :12])).sum(2).min(1)
ST = {'all': np.ones(n, bool), 'tier2': tier == 2, 'tier1': tier == 1, 'glc_all': glc_all, 'glc_not': ~glc_all,
      'complete': nmin == 12, 'n8_11': (nmin >= 8) & (nmin <= 11), 'stable': (tier == 2) & glc_all}
acc = {k: [] for k in ST}
for _ in range(REPS):
    p = np.argsort(rng.random(12)); r = PR.run(X[:, :, p], P[:, p], C, 'shrunk'); I, T, Rk, Imp = PR.flags(r['V'], r['incl']); R = Rk.any(1)
    for k, m in ST.items(): acc[k].append(100 * float(R[m & r['incl']].mean()))
json.dump({'reps': REPS, 'R_by_stratum': {k: dict(mean=float(np.mean(v)), q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975))) for k, v in acc.items()}},
          open(O + '/perm_strata_shrunk.json', 'w'), indent=1)
print('done %.0f s' % (time.time() - t0))
