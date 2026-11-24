/**** MITT-CDA · 04_sample_export.js ******************************************
 * Random sample of cropland pixels (seed 20261111) for offline analysis.
 * Exports:
 *   - adjusted anomalies 2013-2024 (and 2025-2026 when HOLD_ASSET_READY = true);
 *   - stored product flags and Mann-Kendall statistics from MITTCDA_v2_2013_2024;
 *   - coordinates, GAUL district code, cropland mask confidence tier;
 *   - annual GLC_FCS30D classes for 2013-2022;
 *   - incomplete time series retained (dropNulls: false; empty = masked year);
 *   - ERA5-Land covariates (P, PPRE, T2M) exported as a separate 0.1 degree GeoTIFF.
 *
 * Run -> verify GLC_FCS30D band names in Console -> start export tasks.
 *****************************************************************************/

var HOLD_ASSET_READY = false;       // set true after A_holdout_2025_2026 is exported
var HOLD = [2025, 2026];
var SAMPLE = {numPixels: 100000, seed: 20261111, tileScale: 16};
var ROOT = 'projects/akmola/assets/';
var FOLDER = 'akmola_masks';

var A = ee.Image(ROOT + 'A_stack_v2_2013_2024');
var M = ee.Image(ROOT + 'MITTCDA_v2_2013_2024');
var maskAsset = ee.Image(ROOT + 'cropland_mask_v1');
var aoi = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Kazakhstan'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya')).geometry();
var adm2 = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya'));

var IDX = ['NDVI', 'NDMI', 'LST', 'TGSI', 'SI'];
var YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var ALLY = YEARS.concat(HOLD);
var aBands = [];
IDX.forEach(function (k) { YEARS.forEach(function (y) { aBands.push(k + '_' + y); }); });
var mBands = ['severity', 'mechanism', 'T_NDVI', 'T_NDMI', 'T_LST', 'T_TGSI', 'T_SI',
  'NDVI_beta', 'NDMI_beta', 'LST_beta', 'TGSI_beta', 'SI_beta',
  'NDVI_tau', 'NDMI_tau', 'LST_tau', 'TGSI_tau', 'SI_tau'];

// ---------- auxiliary layers (unmasked across the region)
var tier = maskAsset.select('confidence').unmask(0).round().toInt().rename('tier');
var district = ee.Image().toInt().paint(adm2, 'ADM2_CODE').unmask(0).rename('adm2_code');
var lonlat = ee.Image.pixelLonLat().rename(['lon', 'lat']);
print('GAUL district codes and names', adm2.reduceColumns(ee.Reducer.toList(2), ['ADM2_CODE', 'ADM2_NAME']));

// GLC_FCS30D annual maps (2000-2022): bands b1..b23, where 2013 = b14.
var glcAll = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual').filterBounds(aoi).mosaic();
print('GLC_FCS30D bands (expected b1..b23)', glcAll.bandNames());
var glcYears = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022];
var glc = glcAll.select(glcYears.map(function (y) { return 'b' + (y - 1999); }),
  glcYears.map(function (y) { return 'GLC_' + y; })).unmask(0).toInt();

// annual ERA5-Land covariates
var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
function eraYear(y) {
  var win = era.filterDate(ee.Date.fromYMD(y, 6, 15), ee.Date.fromYMD(y, 8, 15));
  var pre = era.filterDate(ee.Date.fromYMD(y, 4, 15), ee.Date.fromYMD(y, 6, 15));
  return ee.Image.cat([
    win.select('total_precipitation_sum').sum().multiply(1000).rename('P_' + y),
    pre.select('total_precipitation_sum').sum().multiply(1000).rename('PPRE_' + y),
    win.select('temperature_2m').mean().subtract(273.15).rename('T2M_' + y)
  ]);
}
var eraStack = ee.Image.cat(ALLY.map(eraYear)).unmask(-9999).toFloat();
var eraBands = [];
ALLY.forEach(function (y) { eraBands.push('P_' + y, 'PPRE_' + y, 'T2M_' + y); });

// ---------- stack and sample (band A_stack_v2 comes first so sampling uses its native grid)
var stack = A.select(aBands).addBands(M.select(mBands)).addBands(tier).addBands(district)
  .addBands(lonlat).addBands(glc);
var cols = aBands.concat(mBands, ['tier', 'adm2_code', 'lon', 'lat'],
  glcYears.map(function (y) { return 'GLC_' + y; }));
// ERA5-Land is exported separately on its native 0.1 degree grid and joined by coordinates
// offline, avoiding per-point sampling timeouts across 42 bands.
Export.image.toDrive({
  image: eraStack, description: 'era5land_covariates_2013_' + HOLD[HOLD.length - 1], folder: FOLDER,
  region: aoi.bounds(1), crs: 'EPSG:4326', crsTransform: [0.1, 0, 65.05, 0, -0.1, 53.05], maxPixels: 1e8
});

if (HOLD_ASSET_READY) {
  var hBands = [];
  IDX.forEach(function (k) { HOLD.forEach(function (y) { hBands.push(k + '_' + y); }); });
  stack = stack.addBands(ee.Image(ROOT + 'A_holdout_' + HOLD.join('_')).select(hBands));
  cols = cols.concat(hBands);
}

var samp = stack.sample({
  region: aoi, numPixels: SAMPLE.numPixels, seed: SAMPLE.seed,
  dropNulls: false, tileScale: SAMPLE.tileScale, geometries: false
}).filter(ee.Filter.notNull(['severity']));       // keep published analysis area only

Export.table.toDrive({
  collection: samp, description: 'mittcda_sample_v3' + (HOLD_ASSET_READY ? '_holdout' : ''),
  folder: FOLDER, fileFormat: 'CSV', selectors: cols
});
print('Table columns', cols.length);
