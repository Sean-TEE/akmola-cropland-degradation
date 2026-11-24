#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures 1, 3, 5, 6 and 7 of the MITT-CDA manuscript.
Usage: python make_figures_main.py RESULTS_DIR OUT_DIR (reads numbers.json, maps.json, scale_ratio.npz, perm_shrunk.json)"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from make_figures import canvas, box, arrow, style, SURF, INK, INK2, GRID, COL, MRK, SEQ

RES, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True); NUM = json.load(open(RES + '/numbers.json')); MAPS = json.load(open(RES + '/maps.json')); RAT = np.load(RES + '/scale_ratio.npz')
PS = json.load(open(RES + '/perm_shrunk.json'))
L2 = NUM['shrunk_scale']; ho = NUM['holdout']
fmt = lambda v: '{:,.0f}'.format(v)

def fig_method():
    fig, ax = canvas(11, 12.0, (0, 100), (-29.5, 81))
    ax.text(50, 78.6, 'Data sources (public archives, Google Earth Engine)', ha='center', va='center', fontsize=11, fontweight='bold')
    S = '#dfe5ee'; P = '#e8edf2'
    box(ax, 3, 25, 66.5, 76, 'Landsat 8 OLI/TIRS', ['Collection 2 Level-2', '666 scenes, 2013–2024', '121 scenes, 2025–2026'], S)
    box(ax, 27.5, 49.5, 66.5, 76, 'ESA WorldCover v200', ['ESA WorldCereal 2021', 'cropland products, 10 m'], S)
    box(ax, 52, 73, 66.5, 76, 'ERA5-Land', ['daily aggregates:', 'total precipitation'], S)
    box(ax, 76, 97, 66.5, 76, 'FAO GAUL 2015', ['administrative boundary', 'of the oblast, level 1 (step 1)'], S)
    box(ax, 3, 25, 53, 63, 'Preprocessing', ['QA_PIXEL cloud mask', 'scale factors; LST > 56 °C removed', 'median composite, 15 Jun – 15 Aug'], P, badge=2)
    box(ax, 27.5, 49.5, 53, 63, 'Cropland mask', ['57,694 km²', 'two confidence tiers'], P, badge=1)
    box(ax, 52, 73, 53, 63, 'Precipitation', ['ERA5-Land total, summed over', 'the 15 Jun – 15 Aug window'], P, badge=2)
    box(ax, 3, 49.5, 42, 50, 'Five mechanism-mapped indicators', ['annual series 2013–2024:  NDVI · NDMI · LST · TGSI · SI', '(BSI excluded: r = 0.95 with NDMI)'], P, badge=3)
    box(ax, 3, 73, 30.5, 39, 'Precipitation-adjusted anomaly module',
        ['baseline 2013–2019:   μ = median,   $\\tilde{σ}$ = 1.4826 · MAD shrunk towards the 25 km cell,   $Z_k = (x_k − μ_k)/\\tilde{σ}_k$',
         '$A_k = δ_k (Z_k − b_k Z_P)$;   $δ_k$ = degradation direction,   $b_k$ from a regional regression'], '#fdefd3', badge=4)
    box(ax, 3, 73, 19, 27.5, 'Local relative normalization',
        ['$M_k$ = median of $A_k$ over the cropland of the 25 km cell in the same year',
         'relative anomaly  $A^*_k = A_k − M_k$'], '#e3f0e1', badge=5)
    box(ax, 3, 36, 5.5, 16, 'Threshold branch (state)', ['evaluation years 2020–2024', '$A^*_k$ > 1.5 in at least 3 of 5 years', 'mechanism signatures D1–D4'], '#f8d9c4', badge=6)
    box(ax, 40, 73, 5.5, 16, 'Trend branch (change)', ['all years 2013–2024', 'Kendall τ ≥ critical value for the', 'series length; Theil–Sen slope > 0'], '#cfe0ee', badge=7)
    box(ax, 3, 36, -2.5, 2.5, '', ['Layer T: %s km² (%.1f%%)' % (fmt(L2['T_km2']), L2['T'])], '#f8d9c4', size=10)
    box(ax, 40, 73, -2.5, 2.5, '', ['Layer R: %s km² (%.1f%%)' % (fmt(L2['R_km2']), L2['R'])], '#cfe0ee', size=10)
    box(ax, 3, 73, -13.5, -5.5, 'Agreement analysis and benchmarks',
        ['T ∩ R: %s km² (%.1f%%)    T ∪ R: %s km² (%.1f%%)    J = %.2f' % (fmt(L2['TR_km2']), L2['TR'], fmt(L2['U_km2']), L2['U'], L2['J']),
         'layer sizes compared with the same procedure on permuted years'], '#e5dcee', badge=8)
    o = PS['observed']['holdout']; w = PS['whole_14']
    box(ax, 3, 73, -27, -16.5, 'Hold-out evaluation, 2025–2026',
        ['baseline, scale and flags of 2013–2024 applied unchanged; reference of the cell from the hold-out year itself;',
         'share of T above 1.5 in 2025: %.0f%%, expected from the same procedure on permuted years %.0f%%, not flagged %.0f%%'
         % (o['T|0'], w['T|0']['mean'], o['none|0'])], '#e3f0e1', badge=9)
    box(ax, 76, 97, -27, 16, 'Persistent assets', ['baseline_stats_2013_2019', '(20 bands: μ, σ, n of years)', '', 'A_stack_v2 (60 bands)', '',
                                                   'A_holdout_2025_2026 (10 bands)', '', 'MITTCDA_v3_grid_25km_r2', '(25 bands, per-pixel scale;', 'not the sample layers)'], '#dddddd', size=8.6)
    for x in (14, 38.5, 62.5):
        arrow(ax, x, 66.5, x, 63)
    arrow(ax, 14, 53, 14, 50); arrow(ax, 38.5, 53, 38.5, 50); arrow(ax, 62.5, 53, 62.5, 39)
    arrow(ax, 26, 42, 26, 39); arrow(ax, 38, 30.5, 38, 27.5)
    arrow(ax, 19.5, 19, 19.5, 16); arrow(ax, 56.5, 19, 56.5, 16)
    arrow(ax, 19.5, 5.5, 19.5, 2.5); arrow(ax, 56.5, 5.5, 56.5, 2.5)
    arrow(ax, 19.5, -2.5, 19.5, -5.5); arrow(ax, 56.5, -2.5, 56.5, -5.5); arrow(ax, 38, -13.5, 38, -16.5)
    arrow(ax, 73, -9.5, 76, -9.5); arrow(ax, 76, -21.75, 73, -21.75)
    fig.savefig(O + '/fig1_method.png', dpi=300, facecolor='white'); plt.close(fig)

def fig_vmodel():
    fig, ax = canvas(11, 6.9, (0, 100), (-1, 68))
    ax.text(21, 65.5, 'Specification', ha='center', va='center', fontsize=11.5, fontweight='bold')
    ax.text(78, 65.5, 'Verification and evaluation', ha='center', va='center', fontsize=11.5, fontweight='bold')
    L, R_ = '#e8edf2', '#cfe0ee'
    box(ax, 58, 98, 54, 63, 'Ground validation', ['field verification of a stratified sample of flagged land', '(next stage, outside the scope of this paper)'], '#ffffff', ec='#777777', lw=1.2, size=8.6, ls=(0, (5, 3)))
    box(ax, 58, 98, 41, 50.5, 'Hold-out evaluation', ['flag groups in 2025 and 2026 compared with', 'the same procedure on permuted years'], R_, size=8.6)
    box(ax, 2, 40, 41, 50.5, 'Requirements', ['management needs; degradation', 'mechanisms of the steppe (D1–D4)'], L, size=8.6)
    box(ax, 55, 95, 28, 37.5, 'System verification', ['2.9–5.5 clear observations per pixel;', 'ERA5-Land 90 mm vs. stations 90.5 mm;', 'layer sizes against permutation benchmarks'], R_, size=8.6)
    box(ax, 5, 43, 28, 37.5, 'System design', ['five mechanism-mapped indicators;', 'local relative normalization; two decision branches'], L, size=8.6)
    box(ax, 52, 92, 15, 24.5, 'Module verification', ['baseline Z centred (|mean| ≤ 0.11);', 'pixel scale vs. cell scale (selection check);', 're-run: anomalies within 0.013 σ at five points'], R_, size=8.6)
    box(ax, 8, 46, 15, 24.5, 'Detailed design', ['robust baseline, scale shrunk to the 25 km cell;', 'precipitation adjustment; 25 km reference;', '1.5 σ in 3 of 5 years; Kendall τ by series length'], L, size=8.6)
    box(ax, 24, 76, 2, 11.5, 'Implementation', ['Google Earth Engine · 666 + 121 Landsat 8 scenes · 57,663 km² analysis area'], '#fdefd3', size=8.6)
    arrow(ax, 21, 41, 24, 37.5); arrow(ax, 24, 28, 27, 24.5); arrow(ax, 30, 15, 36, 11.5)
    arrow(ax, 64, 11.5, 70, 15); arrow(ax, 72, 24.5, 75, 28); arrow(ax, 75, 37.5, 78, 41)
    for (x0, x1, yy) in ((40, 58, 45.75), (43, 55, 32.75), (46, 52, 19.75)):
        arrow(ax, x0 + 0.5, yy, x1 - 0.5, yy, color='#777777', ls=(0, (5, 3)))
    arrow(ax, 78, 50.5, 78, 54, color='#777777', ls=(0, (5, 3)))
    fig.savefig(O + '/fig3_vmodel.png', dpi=300, facecolor='white'); plt.close(fig)

def fig_maps():
    cs = 12.5; cx = np.array(MAPS['cx']) / 1000; cy = np.array(MAPS['cy']) / 1000; cnt = np.array(MAPS['count']); ok = cnt >= 15
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), facecolor=SURF)
    cols = (('uncorrected', 'Uncorrected'), ('pixel_scale', 'Local normalization, pixel scale'), ('shrunk_scale', 'Local normalization, shrunk scale'))
    for r_, (lay, ln) in enumerate((('T', 'threshold layer T'), ('R', 'trend layer R'))):
        for c_, (key, title) in enumerate(cols):
            ax = axs[r_, c_]; z = np.array(MAPS['%s_%s' % (lay, key)]); ax.set_facecolor(SURF)
            for xx, yy, v in zip(cx[ok], cy[ok], z[ok]):
                ax.add_patch(Rectangle((xx - cs / 2, yy - cs / 2), cs, cs, fc=SEQ(min(v / 0.5, 1.0)), ec=SURF, lw=0.3))
            ax.set_xlim(cx[ok].min() - 10, cx[ok].max() + 10); ax.set_ylim(cy[ok].min() - 12, cy[ok].max() + 10)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title('%s\n%s: %.1f%% of the sample' % (title, ln, NUM[key][lay]), loc='left', fontsize=9.5, pad=4)
            x0 = cx[ok].min() - 6; y0 = cy[ok].min() - 9
            ax.plot([x0, x0 + 100], [y0, y0], color=INK, lw=2.2, solid_capstyle='butt'); ax.text(x0 + 50, y0 + 3, '100 km', ha='center', va='bottom', fontsize=7.5, color=INK2)
    cax = fig.add_axes([0.33, 0.07, 0.34, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 50), cmap=SEQ), cax=cax, orientation='horizontal', extend='max')
    cb.set_label('Share of sampled cropland pixels flagged in a 12.5 km cell, %', fontsize=9, color=INK2); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.13, wspace=0.03, hspace=0.18)
    fig.savefig(O + '/fig5_maps.png', dpi=300, facecolor=SURF); plt.close(fig)

def fig_diag():
    YA = list(range(2013, 2027))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.9), facecolor=SURF, gridspec_kw={'width_ratios': [1.55, 1]})
    ax = axs[0]; style(ax)
    ax.axvspan(2019.5, 2024.5, color='#f0efec', zorder=0, lw=0); ax.axvspan(2024.5, 2026.5, color='#e6ecf5', zorder=0, lw=0)
    for j, (key, lab) in enumerate((('uncorrected', 'uncorrected'), ('pixel_scale', 'local normalization, pixel scale'), ('shrunk_scale', 'local normalization, shrunk scale'))):
        v = NUM[key]['exceed_NDVI_by_year']
        ax.plot(YA, v, color=COL[j], lw=2, marker=MRK[j], ms=5.5, mec=SURF, mew=1, label=lab, zorder=3)
    ax.set_xticks(YA); ax.set_xticklabels([str(v) for v in YA], fontsize=8.5); ax.set_xlim(2012.5, 2026.5); ax.set_ylim(0, 42)
    ax.set_ylabel('Sample pixels with NDVI anomaly above 1.5, %')
    ax.set_title('(a) Exceedance of the threshold by year', loc='left', fontsize=11, pad=8)
    ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK2, handlelength=2.2)
    ax.text(2016, 1, 'baseline', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.text(2022, 1, 'evaluation period', ha='center', va='bottom', fontsize=8.5, color=INK2); ax.text(2025.5, 1, 'hold-out', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax = axs[1]; style(ax); bins = np.linspace(-3, 2, 26)
    for arr, lab, c in ((RAT['N0'], 'not flagged', '#9c9a93'), (RAT['T0'], 'layer T, pixel scale', COL[1]), (RAT['T2'], 'layer T, shrunk scale', COL[2])):
        a = np.log2(arr[(arr > 0) & np.isfinite(arr)]); h, e = np.histogram(a, bins); h = h / h.sum() * 100
        ax.step(e[:-1], h, where='post', color=c, lw=2, label='%s (median %.2f)' % (lab, np.median(arr[arr > 0])))
    ax.axvline(0, color='#b9b8b2', lw=1)
    ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_xticklabels(['1/8', '1/4', '1/2', '1', '2', '4'])
    ax.set_xlabel('Scale of the pixel / median scale of its 25 km cell (NDVI)'); ax.set_ylabel('Share of the group, %')
    ax.set_title('(b) Baseline scale of the flagged pixels', loc='left', fontsize=11, pad=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5, labelcolor=INK2, handlelength=1.6)
    fig.tight_layout(); fig.savefig(O + '/fig6_scale_diagnosis.png', dpi=300, facecolor=SURF); plt.close(fig)

def fig_holdout():
    YA = list(range(2013, 2027))
    groups = (('T ∩ R', 'T and R', 'T and R'), ('threshold only', 'T only', 'T only'), ('trend only', 'R only', 'R only own axes'), ('not flagged', 'none', 'none'))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.9), facecolor=SURF, gridspec_kw={'width_ratios': [1.55, 1]})
    ax = axs[0]; style(ax)
    ax.axvspan(2019.5, 2024.5, color='#f0efec', zorder=0, lw=0); ax.axvspan(2024.5, 2026.5, color='#e6ecf5', zorder=0, lw=0)
    ax.axhline(0, color='#b9b8b2', lw=1, zorder=1); ax.axhline(1.5, color='#b9b8b2', lw=1, zorder=1)
    for j, (lab, g, _) in enumerate(groups):
        ax.plot(YA, ho[g]['traj'], color=COL[j], lw=2, marker=MRK[j], ms=5.5, mec=SURF, mew=1, label=lab, zorder=3)
    ax.set_xticks(YA); ax.set_xticklabels([str(v) for v in YA], fontsize=8.5); ax.set_xlim(2012.5, 2026.5)
    ax.set_ylabel('Median relative NDVI anomaly, σ')
    ax.set_title('(a) Median relative NDVI anomaly of the four flag groups', loc='left', fontsize=11, pad=8)
    ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK2, handlelength=2.2)
    yl = ax.get_ylim(); ax.text(2022, yl[0] + 0.04, 'evaluation period', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.text(2025.5, yl[0] + 0.04, 'hold-out', ha='center', va='bottom', fontsize=8.5, color=INK2); ax.text(2012.7, 1.54, 'threshold 1.5', fontsize=8.5, color=INK2, va='bottom')
    ax = axs[1]; style(ax); o = PS['observed']['holdout']; w = PS['whole_14']; wbar = 0.34
    for j, (lab, _, key) in enumerate(groups):
        for i in range(2):
            xx = j + (i - 0.5) * (wbar + 0.04); v = o['%s|%d' % (key, i)]; b = w['%s|%d' % (key, i)]
            ax.bar(xx, v, width=wbar, color=COL[j], alpha=1.0 if i == 0 else 0.55, zorder=3)
            ax.plot([xx, xx], [b['q025'], b['q975']], color=INK, lw=1.4, zorder=4); ax.plot([xx - 0.09, xx + 0.09], [b['mean'], b['mean']], color=INK, lw=1.4, zorder=4)
            ax.text(xx, max(v, b['q975']) + 1.2, '%.0f' % v, ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(['T ∩ R', 'threshold\nonly', 'trend only\n(own axis)', 'not\nflagged'], fontsize=8.5)
    ax.set_ylabel('Pixels above 1.5 in the hold-out year, %'); ax.set_ylim(0, 66)
    ax.set_title('(b) Hold-out exceedance: 2025 (dark), 2026 (light)', loc='left', fontsize=11, pad=8)
    ax.text(3.45, 62, 'black: same procedure on permuted years,\nmean and 95% range', ha='right', va='top', fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(O + '/fig7_holdout.png', dpi=300, facecolor=SURF); plt.close(fig)

fig_method(); fig_vmodel(); fig_maps(); fig_diag(); fig_holdout(); print('ok')
