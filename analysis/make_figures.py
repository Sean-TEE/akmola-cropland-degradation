#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Year-effect figure (Figure 4) and shared drawing helpers for MITT-CDA. Usage: python make_figures.py DATA_DIR OUT_DIR"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import rasterio
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mittcda_v3_analysis as M
warnings.filterwarnings('ignore')

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MRK = ['o', 's', '^', 'D', 'v']
SEQ = LinearSegmentedColormap.from_list('blue', ['#eef4fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})

def style(ax):
    ax.set_facecolor(SURF); ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

# ------------------------------------------------------------------ diagrams
def canvas(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h), facecolor='white'); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
    ax._ux = w * 72 / (xlim[1] - xlim[0]); ax._uy = h * 72 / (ylim[1] - ylim[0])      # points per data unit
    return fig, ax

def fit_text(ax, x, y, s, maxw, size, **kw):
    """Draw centred text and shrink the font until the text is not wider than maxw data units."""
    t = ax.text(x, y, s, ha='center', va='center', fontsize=size, zorder=3, **kw)
    rend = ax.figure.canvas.get_renderer()
    wpt = t.get_window_extent(rend).width * 72 / ax.figure.dpi
    if wpt > maxw * ax._ux:
        t.set_fontsize(size * maxw * ax._ux / wpt * 0.99)
    return t

def box(ax, x0, x1, y0, y1, title, lines, fc, ec='#222222', lw=1.1, tsize=10.5, size=9.0, badge=None, ls='-'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.9', fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ht = 1.5 * tsize / ax._uy if title else 0.0; hl = 1.42 * size / ax._uy
    total = ht + hl * len(lines); top = (y0 + y1) / 2 + total / 2; xc = (x0 + x1) / 2
    if title:
        fit_text(ax, xc, top - ht / 2, title, (x1 - x0) - (7.0 if badge is not None else 2.0), tsize, fontweight='bold')
    for i, ln in enumerate(lines):
        if ln:
            fit_text(ax, xc, top - ht - hl * (i + 0.5), ln, (x1 - x0) - 2.0, size)
    if badge is not None:
        ax.scatter([x0 + 0.3], [y1 - 0.3], s=520, c='#222222', zorder=4, edgecolors='white', linewidths=1.2)
        ax.text(x0 + 0.3, y1 - 0.38, str(badge), ha='center', va='center', color='white', fontsize=10.5, fontweight='bold', zorder=5)

def arrow(ax, x0, y0, x1, y1, color='#333333', ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=11))

def fig_method(out):
    fig, ax = canvas(11, 12.0, (0, 100), (-29.5, 81))
    ax.text(50, 78.6, 'Data sources (public archives, Google Earth Engine)', ha='center', va='center', fontsize=11, fontweight='bold')
    S = '#dfe5ee'; P = '#e8edf2'
    box(ax, 3, 25, 66.5, 76, 'Landsat 8 OLI/TIRS', ['Collection 2 Level-2', '666 scenes, 2013–2024', '121 scenes, 2025–2026'], S)
    box(ax, 27.5, 49.5, 66.5, 76, 'ESA WorldCover v200', ['ESA WorldCereal 2021', 'cropland products, 10 m'], S)
    box(ax, 52, 73, 66.5, 76, 'ERA5-Land', ['daily aggregates:', 'total precipitation'], S)
    box(ax, 76, 97, 66.5, 76, 'FAO GAUL 2015', ['oblast boundary (step 1)', 'district units (step 8)'], S)
    box(ax, 3, 25, 53, 63, 'Preprocessing', ['QA_PIXEL cloud mask', 'scale factors; LST > 56 °C removed', 'median composite, 15 Jun – 15 Aug'], P, badge=2)
    box(ax, 27.5, 49.5, 53, 63, 'Cropland mask', ['57,694 km²', 'two confidence tiers'], P, badge=1)
    box(ax, 52, 73, 53, 63, 'Precipitation', ['ERA5-Land total, summed over', 'the 15 Jun – 15 Aug window'], P, badge=2)
    box(ax, 3, 49.5, 42, 50, 'Five mechanism-mapped indicators', ['annual series 2013–2024:  NDVI · NDMI · LST · TGSI · SI', '(BSI excluded: r = 0.95 with NDMI)'], P, badge=3)
    box(ax, 3, 73, 30.5, 39, 'Precipitation-adjusted anomaly module',
        ['baseline 2013–2019:   μ = median,   σ = 1.4826 · MAD,   $Z_k = (x_k − μ_k)/σ_k$',
         '$A_k = δ_k (Z_k − b_k Z_P)$;   $δ_k$ = degradation direction,   $b_k$ from a regional regression'], '#fdefd3', badge=4)
    box(ax, 3, 73, 19, 27.5, 'Removal of the local year effect',
        ['$M_k$ = median of $A_k$ over cropland in 25 km cells, interpolated between cell centres',
         'relative anomaly  $A^*_k = A_k − M_k$'], '#e3f0e1', badge=5)
    box(ax, 3, 36, 5.5, 16, 'Threshold branch (state)', ['evaluation years 2020–2024', '$A^*_k$ > 1.5 in at least 3 of 5 years', 'mechanism signatures D1–D4'], '#f8d9c4', badge=6)
    box(ax, 40, 73, 5.5, 16, 'Trend branch (change)', ['all years 2013–2024', 'Kendall τ ≥ critical value for the', 'series length; Theil–Sen slope > 0'], '#cfe0ee', badge=7)
    box(ax, 3, 36, -2.5, 2.5, '', ['Layer T: 3,036 km² (5.3%)'], '#f8d9c4', size=10)
    box(ax, 40, 73, -2.5, 2.5, '', ['Layer R: 7,976 km² (13.8%)'], '#cfe0ee', size=10)
    box(ax, 3, 73, -13.5, -5.5, 'Agreement analysis', ['T ∩ R: 1,374 km² (2.4%)      T ∪ R: 9,638 km² (16.7%)      Jaccard J = 0.14'], '#e5dcee', badge=8)
    box(ax, 3, 73, -27, -16.5, 'Hold-out evaluation, 2025–2026',
        ['anomalies of two years that were not used for the baseline, the reference or the flags;',
         'share of pixels above 1.5 in 2025: T ∩ R 44%, T only 33%, R only 21%, not flagged 14%'], '#e3f0e1', badge=9)
    box(ax, 76, 97, -27, 16, 'Persistent assets', ['MITTCDA_v3_grid_25km (25 bands):', 'DI, DI_thr, DI_trend,', 'severity, mechanism,',
                                                   'per-axis flag, slope, τ, n', '', 'A_stack_v2 (60 bands)', '', 'A_holdout_2025_2026 (10 bands)'], '#dddddd', size=8.6)
    for x in (14, 38.5, 62.5):
        arrow(ax, x, 66.5, x, 63)
    arrow(ax, 14, 53, 14, 50); arrow(ax, 38.5, 53, 38.5, 50); arrow(ax, 62.5, 53, 62.5, 39)
    arrow(ax, 26, 42, 26, 39); arrow(ax, 38, 30.5, 38, 27.5)
    arrow(ax, 19.5, 19, 19.5, 16); arrow(ax, 56.5, 19, 56.5, 16)
    arrow(ax, 19.5, 5.5, 19.5, 2.5); arrow(ax, 56.5, 5.5, 56.5, 2.5)
    arrow(ax, 19.5, -2.5, 19.5, -5.5); arrow(ax, 56.5, -2.5, 56.5, -5.5); arrow(ax, 38, -13.5, 38, -16.5)
    arrow(ax, 73, -9.5, 76, -9.5); arrow(ax, 76, -21.75, 73, -21.75)
    fig.savefig(out + '/fig1_method.png', dpi=300, facecolor='white'); plt.close(fig)

def fig_vmodel(out):
    fig, ax = canvas(11, 6.9, (0, 100), (-1, 68))
    ax.text(21, 65.5, 'Specification', ha='center', va='center', fontsize=11.5, fontweight='bold')
    ax.text(78, 65.5, 'Verification and evaluation', ha='center', va='center', fontsize=11.5, fontweight='bold')
    L, R_ = '#e8edf2', '#cfe0ee'
    box(ax, 58, 98, 54, 63, 'Ground validation', ['field verification of a stratified sample of flagged land', '(next stage, outside the scope of this paper)'], '#ffffff', ec='#777777', lw=1.2, size=8.6, ls=(0, (5, 3)))
    box(ax, 58, 98, 41, 50.5, 'Hold-out evaluation', ['flag groups compared in 2025 and 2026,', 'two years not used for calibration'], R_, size=8.6)
    box(ax, 2, 40, 41, 50.5, 'Requirements', ['management needs; degradation', 'mechanisms of the steppe (D1–D4)'], L, size=8.6)
    box(ax, 55, 95, 28, 37.5, 'System verification', ['2.9–5.5 clear observations per pixel;', 'ERA5-Land 90 mm vs. stations 90.5 mm;', 'full raster within 0.2 pp of the sample'], R_, size=8.6)
    box(ax, 5, 43, 28, 37.5, 'System design', ['five mechanism-mapped indicators;', 'local year-effect reference; two decision branches'], L, size=8.6)
    box(ax, 52, 92, 15, 24.5, 'Module verification', ['baseline Z centred (|mean| ≤ 0.11);', 'σ floors bind on 0.1–3.5% of cropland;', 're-run reproduces anomalies within 0.013 σ'], R_, size=8.6)
    box(ax, 8, 46, 15, 24.5, 'Detailed design', ['robust baseline; precipitation adjustment;', '25 km reference; 1.5 σ in 3 of 5 years;', 'Kendall τ by series length'], L, size=8.6)
    box(ax, 24, 76, 2, 11.5, 'Implementation', ['Google Earth Engine · 666 + 121 Landsat 8 scenes · 57,663 km² analysis area'], '#fdefd3', size=8.6)
    arrow(ax, 21, 41, 24, 37.5); arrow(ax, 24, 28, 27, 24.5); arrow(ax, 30, 15, 36, 11.5)
    arrow(ax, 64, 11.5, 70, 15); arrow(ax, 72, 24.5, 75, 28); arrow(ax, 75, 37.5, 78, 41)
    for (x0, x1, yy) in ((40, 58, 45.75), (43, 55, 32.75), (46, 52, 19.75)):
        arrow(ax, x0 + 0.5, yy, x1 - 0.5, yy, color='#777777', ls=(0, (5, 3)))
    arrow(ax, 78, 50.5, 78, 54, color='#777777', ls=(0, (5, 3)))
    fig.savefig(out + '/fig3_vmodel.png', dpi=300, facecolor='white'); plt.close(fig)

# ------------------------------------------------------------------ data figures
def main():
    D, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
    fig_method(O); fig_vmodel(O)
    df, A, H, x, y, ref_reg, ref, ncell = M.load(D); As = A - ref
    T2 = M.signature(M.active(A)); R2 = M.trend_flags(A)[0].any(1); T3 = M.signature(M.active(As)); R3 = M.trend_flags(As)[0].any(1)
    r = rasterio.open(D + '/era5land_covariates_2013_2026.tif'); E = r.read().astype(float); names = list(r.descriptions)
    col = np.floor((df.lon.values - r.transform.c) / r.transform.a).astype(int); row = np.floor((df.lat.values - r.transform.f) / r.transform.e).astype(int)
    ppre = np.array([E[names.index(f'PPRE_{yy}')][row, col].mean() for yy in range(2013, 2027)])
    YA = list(range(2013, 2027)); hold_med = np.nanmedian(H, 0)                      # (5,2)

    # ---- Figure: year effect and pre-season precipitation
    fig, axs = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True, facecolor=SURF, gridspec_kw={'height_ratios': [2.1, 1]})
    ax = axs[0]; style(ax)
    ax.axvspan(2019.5, 2024.5, color='#f0efec', zorder=0, lw=0); ax.axvspan(2024.5, 2026.5, color='#e6ecf5', zorder=0, lw=0)
    ax.axhline(0, color='#b9b8b2', lw=1, zorder=1)
    for i, k in enumerate(M.IDX):
        ax.plot(M.YEARS, ref_reg[i], color=COL[i], lw=2, marker=MRK[i], ms=5.5, mec=SURF, mew=1, label=k, zorder=3)
        ax.plot([2024, 2025, 2026], [ref_reg[i][-1], hold_med[i, 0], hold_med[i, 1]], color=COL[i], lw=1.4, ls=(0, (3, 2)), zorder=2)
        ax.plot([2025, 2026], hold_med[i], color=COL[i], lw=0, marker=MRK[i], ms=6, mfc=SURF, mec=COL[i], mew=1.6, zorder=3)
    ax.set_ylabel('Regional median of the adjusted anomaly,\nrobust σ (positive = worse)')
    ax.set_title('Regional median of the precipitation-adjusted anomaly by axis', loc='left', fontsize=11, pad=8)
    ax.legend(frameon=False, loc='upper left', fontsize=9, ncol=5, handlelength=2, labelcolor=INK2, columnspacing=1.4)
    yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[1] + 0.35)
    ax.text(2022, yl[0] + 0.05, 'evaluation period', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.text(2025.5, yl[0] + 0.05, 'hold-out', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax = axs[1]; style(ax)
    ax.axvspan(2019.5, 2024.5, color='#f0efec', zorder=0, lw=0); ax.axvspan(2024.5, 2026.5, color='#e6ecf5', zorder=0, lw=0)
    ax.bar(YA, ppre, width=0.5, color=COL[0], zorder=3)
    bm = ppre[:7].mean(); ax.axhline(bm, color='#52514e', lw=1, zorder=2)
    ax.text(2016.5, bm + 3, 'baseline mean %.0f mm' % bm, fontsize=8.5, color=INK2, va='bottom', ha='center')
    for yy in (2021, 2022, 2023, 2024):
        ax.text(yy, ppre[yy - 2013] + 3, '%.0f' % ppre[yy - 2013], ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.set_ylabel('Precipitation,\n15 Apr – 14 Jun, mm'); ax.set_ylim(0, ppre.max() + 28)
    ax.set_title('Pre-season precipitation (ERA5-Land, mean over the sampled cropland)', loc='left', fontsize=11, pad=8)
    ax.set_xticks(YA); ax.set_xticklabels([str(v) for v in YA], fontsize=9); ax.set_xlim(2012.4, 2026.6)
    fig.tight_layout(); fig.savefig(O + '/fig_year_effect.png', dpi=300, facecolor=SURF); plt.close(fig)

    # ---- Figure: share of sampled cropland flagged per 12.5 km cell, before and after removal of the year effect
    cs = 12500.0; cx = np.floor(x / cs).astype(int); cy = np.floor(y / cs).astype(int)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.0), facecolor=SURF)
    panels = (('Threshold layer T, before removal of the year effect', T2), ('Threshold layer T, after removal', T3),
              ('Trend layer R, before removal', R2), ('Trend layer R, after removal', R3))
    for ax, (title, z) in zip(axs.ravel(), panels):
        g = pd.DataFrame({'cx': cx, 'cy': cy, 'z': z}).groupby(['cx', 'cy']).z.agg(['mean', 'count']).reset_index(); g = g[g['count'] >= 15]
        ax.set_facecolor(SURF)
        for _, rw in g.iterrows():
            ax.add_patch(Rectangle((rw.cx * cs / 1000, rw.cy * cs / 1000), cs / 1000, cs / 1000, fc=SEQ(min(rw['mean'] / 0.5, 1.0)), ec=SURF, lw=0.4))
        ax.set_xlim(g.cx.min() * cs / 1000 - 5, g.cx.max() * cs / 1000 + 20); ax.set_ylim(g.cy.min() * cs / 1000 - 5, g.cy.max() * cs / 1000 + 20)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title('%s: %.1f%% of the sample' % (title, 100 * z.mean()), loc='left', fontsize=10, pad=6)
        x0 = g.cx.min() * cs / 1000; y0 = g.cy.min() * cs / 1000 - 2
        ax.plot([x0, x0 + 100], [y0, y0], color=INK, lw=2.5, solid_capstyle='butt'); ax.text(x0 + 50, y0 + 5, '100 km', ha='center', va='bottom', fontsize=8, color=INK2)
    cax = fig.add_axes([0.33, 0.085, 0.34, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 50), cmap=SEQ), cax=cax, orientation='horizontal', extend='max')
    cb.set_label('Share of sampled cropland pixels flagged in a 12.5 km cell, %', fontsize=9, color=INK2); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15, wspace=0.03, hspace=0.14)
    fig.savefig(O + '/fig_maps_before_after.png', dpi=300, facecolor=SURF); plt.close(fig)

    # ---- Figure: hold-out evaluation
    cell = np.floor(x / M.CELL).astype(int) * 10000 + np.floor(y / M.CELL).astype(int)
    def rel(v):
        s = pd.Series(v); gm = s.groupby(cell).transform('median'); gc = s.groupby(cell).transform('count')
        return (s - np.where(gc >= 30, gm, np.nanmedian(v))).values
    Hrel = np.stack([rel(H[:, 0, j]) for j in range(2)], 1)
    groups = (('T ∩ R', T3 & R3), ('threshold only', T3 & ~R3), ('trend only', R3 & ~T3), ('not flagged', ~T3 & ~R3))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.9), facecolor=SURF, gridspec_kw={'width_ratios': [1.7, 1]})
    ax = axs[0]; style(ax)
    ax.axvspan(2019.5, 2024.5, color='#f0efec', zorder=0, lw=0); ax.axvspan(2024.5, 2026.5, color='#e6ecf5', zorder=0, lw=0)
    ax.axhline(0, color='#b9b8b2', lw=1, zorder=1); ax.axhline(1.5, color='#b9b8b2', lw=1, zorder=1)
    for j, (lab, m) in enumerate(groups):
        tr = np.r_[np.nanmedian(As[m, 0, :], 0), np.nanmedian(Hrel[m], 0)]
        ax.plot(YA, tr, color=COL[j], lw=2, marker=MRK[j], ms=5.5, mec=SURF, mew=1, label=lab, zorder=3)
    ax.set_xticks(YA); ax.set_xticklabels([str(v) for v in YA], fontsize=8.5); ax.set_xlim(2012.5, 2026.5)
    ax.set_ylabel('Median relative NDVI anomaly, robust σ')
    ax.set_title('Median relative NDVI anomaly of the four flag groups', loc='left', fontsize=11, pad=8)
    ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK2, handlelength=2.2)
    yl = ax.get_ylim()
    ax.text(2022, yl[0] + 0.04, 'evaluation period', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.text(2025.5, yl[0] + 0.04, 'hold-out', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.text(2012.7, 1.54, 'threshold 1.5', fontsize=8.5, color=INK2, va='bottom')
    ax = axs[1]; style(ax); wbar = 0.17
    for j, (lab, m) in enumerate(groups):
        for i in range(2):
            v = Hrel[m, i]; sh = 100 * np.mean(v[~np.isnan(v)] > 1.5); xx = i + (j - 1.5) * (wbar + 0.03)
            ax.bar(xx, sh, width=wbar, color=COL[j], zorder=3, label=lab if i == 0 else None)
            ax.text(xx, sh + 1, '%.0f' % sh, ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['2025', '2026']); ax.set_ylabel('Pixels with relative NDVI anomaly above 1.5, %')
    ax.set_ylim(0, 76); ax.set_title('Exceedance in the hold-out years', loc='left', fontsize=11, pad=8)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5, labelcolor=INK2, ncol=2, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(O + '/fig_holdout.png', dpi=300, facecolor=SURF); plt.close(fig)
    print('figures written')

if __name__ == '__main__':
    main()
