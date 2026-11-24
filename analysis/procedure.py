#!/usr/bin/env python3
"""MITT-CDA on the sample: one procedure from indicator values to flags, used for the observed order of years and for permutations.

The indicator values x of every sample pixel are recovered from the stored adjusted anomalies:  x = mu + sigma * (delta*A + b*Z_P),
with the baseline median mu and the floored scale sigma exported by GEE script 05. The procedure then re-estimates everything that
the method estimates from the baseline positions (median, scale, shrinkage weights, standardized precipitation) and applies the
local reference and the decision rules. The precipitation coefficients b are constants of the method and are not re-estimated."""
import numpy as np, pandas as pd, rasterio
import mittcda_v3_analysis as M
import stepb as S

IDX = M.IDX; DELTA, B, FLOOR, V_EST, MINC = S.DELTA, S.B, S.FLOOR, S.V_EST, S.MINC
YRS = list(range(2013, 2027))

def load(D):
    df, A, sig, raw, nb, cell, ZP, x, y = S.load(D)
    mu = df[['mu_' + k for k in IDX]].values
    r = rasterio.open(D + '/era5land_covariates_2013_2026.tif'); E = r.read().astype(float); names = list(r.descriptions)
    col = np.floor((df.lon.values - r.transform.c) / r.transform.a).astype(int); row = np.floor((df.lat.values - r.transform.f) / r.transform.e).astype(int)
    P = np.stack([E[names.index(f'P_{yy}')][row, col] for yy in YRS], 1)                       # (n,14) precipitation of the window, mm
    Z = DELTA[None, :, None] * A + B[None, :, None] * ZP[:, None, :]
    X = mu[:, :, None] + sig[:, :, None] * Z                                                   # (n,5,14) indicator values
    return dict(df=df, A=A, X=X, P=P, cell=cell, C=S.Cells(cell), x=x, y=y, sig_gee=sig, raw_gee=raw, ZP_gee=ZP)

def zp_of(P):
    m = np.median(P[:, :7], 1, keepdims=True); s = np.maximum(1.4826 * np.median(np.abs(P[:, :7] - m), 1, keepdims=True), 1e-6)
    return (P - m) / s

def run(X, P, C, mode='shrunk', b=None):
    """Return V (n,5,L) relative anomalies, included pixels, the scale used and diagnostics. mode: pixel | shrunk | pooled."""
    b = B if b is None else b
    base = X[:, :, :7]; mu = np.nanmedian(base, -1); raw = 1.4826 * np.nanmedian(np.abs(base - mu[:, :, None]), -1)
    nb = (~np.isnan(base)).sum(-1); incl = nb[:, 0] >= 5                                      # GEE rule: at least five valid baseline years of NDVI
    sc = C.median(raw)
    if mode == 'pixel':
        st = np.maximum(raw, FLOOR[None, :]); w = None
    else:
        lnr = np.where(raw > 0, np.log(np.where(raw > 0, raw, 1.0)), np.nan)
        vobs = C.within_var(lnr)
        vest = np.full(nb.shape, np.inf)
        for k_, v_ in V_EST.items(): vest[nb == k_] = v_
        w = np.zeros_like(raw) if mode == 'pooled' else np.clip(1 - vest / vobs[None, :], 0, 1)
        lnmix = np.where(np.isnan(lnr), np.log(sc), w * np.nan_to_num(lnr) + (1 - w) * np.log(sc))
        st = np.maximum(np.exp(lnmix), FLOOR[None, :])
    Z = (X - mu[:, :, None]) / st[:, :, None]
    Aq = DELTA[None, :, None] * (Z - b[None, :, None] * zp_of(P)[:, None, :])
    V = np.stack([Aq[:, :, j] - C.median(Aq[:, :, j]) for j in range(X.shape[2])], 2)
    V[~incl] = np.nan
    return dict(V=V, incl=incl, raw=raw, sc=sc, st=st, w=w, Aq=Aq)

def flags(V, incl):
    I = M.active(V[:, :, :12]) & incl[:, None]; T = M.signature(I)
    Rk, Imp = M.trend_flags(V[:, :, :12]); Rk &= incl[:, None]; Imp &= incl[:, None]
    return I, T, Rk, Imp
