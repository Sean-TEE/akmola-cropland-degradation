/**** MITT-CDA · 05_baseline_stats.js *****************************************
 * Per-pixel baseline statistics (2013-2019) across the five indicators:
 *   mu_<IDX>      - baseline median;
 *   sigma_<IDX>   - 1.4826 * MAD after applying the lower floor;
 *   sigraw_<IDX>  - 1.4826 * MAD before applying the lower floor;
 *   nbase_<IDX>   - number of valid baseline years (5 to 7).
 *
 * Used to recover raw indicator values from the exported anomalies and compute
 * the neighbourhood-shrunk scale in the Python analysis scripts.
 *
 * Two-step workflow:
 *   1. STATS_ASSET_READY = false -> Run -> start baseline_stats_2013_2019 asset export.
 *   2. Once the asset is ready: STATS_ASSET_READY = true -> Run -> start
 *      mittcda_sample_v4_sigma CSV export (121 columns, identical pixel order
 *      and coordinates to script 04).
 *****************************************************************************/

var STATS_ASSET_READY = false;      // set true after baseline_stats_2013_2019 is exported
var HOLD = [2025, 2026];
var SAMPLE = {numPixels: 100000, seed: 20261111, tileScale: 16};
var ROOT = 'projects/akmola/assets/';
var FOLDER = 'akmola_masks';
var STATS_ID = ROOT + 'baseline_stats_2013_2019';

var A = ee.Image(ROOT + 'A_stack_v2_2013_2024');
var M = ee.Image(ROOT + 'MITTCDA_v2_2013_2024');
var maskAsset = ee.Image(ROOT + 'cropland_mask_v1');
var crop = maskAsset.select('cropland');
var aoi = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Kazakhstan'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya')).geometry();
var adm2 = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya'));
var P = A.projection().getInfo();

var IDX = ['NDVI', 'NDMI', 'LST', 'TGSI', 'SI'];
var BASE = [2013, 2014, 2015, 2016, 2017, 2018, 2019];
var YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var FLOOR = {NDVI: 0.01, NDMI: 0.01, TGSI: 0.005, SI: 0.002, LST: 0.3};
var STAT_NAMES = [];
['mu_', 'sigma_', 'sigraw_', 'nbase_'].forEach(function (p) { IDX.forEach(function (k) { STAT_NAMES.push(p + k); }); });

if (!STATS_ASSET_READY) {
  // ---------- Landsat 8 C2 L2: cloud mask and composite (same as script 03)
  var maskL8 = function (img) {
    var qa = img.select('QA_PIXEL');
    var clear = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 2).eq(0))
      .and(qa.bitwiseAnd(1 << 3).eq(0)).and(qa.bitwiseAnd(1 << 4).eq(0));
    var optical = img.select('SR_B.').multiply(2.75e-05).add(-0.2);
    var thermal = img.select('ST_B10').multiply(0.00341802).add(149.0);
    var valid = optical.gt(0).and(optical.lt(1)).reduce(ee.Reducer.min());
    return img.addBands(optical, null, true).addBands(thermal, null, true)
      .updateMask(clear).updateMask(valid);
  };
  var scenes = function (year) {
    return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi)
      .filterDate(ee.Date.fromYMD(year, 6, 15), ee.Date.fromYMD(year, 8, 15))
      .filter(ee.Filter.lt('CLOUD_COVER', 60));
  };
  var indicesForYear = function (year) {
    var comp = scenes(year).map(maskL8).median().clip(aoi);
    var ndvi = comp.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
    var ndmi = comp.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI');
    var lst = comp.select('ST_B10').subtract(273.15).rename('LST');
    lst = lst.updateMask(lst.lte(56));
    var tgsi = comp.expression('(RED-BLUE)/(RED+BLUE+GREEN)', {
      RED: comp.select('SR_B4'), BLUE: comp.select('SR_B2'), GREEN: comp.select('SR_B3')
    }).rename('TGSI').clamp(-1, 1);
    var si = comp.expression('sqrt(BLUE*RED)', {
      BLUE: comp.select('SR_B2'), RED: comp.select('SR_B4')
    }).rename('SI');
    return ee.Image.cat([ndvi, ndmi, lst, tgsi, si]).updateMask(crop).set('year', year);
  };

  // ---------- baseline statistics (same as script 03)
  var baseColl = ee.ImageCollection(BASE.map(indicesForYear)).select(IDX);
  var mu = baseColl.median();
  var sigRaw = baseColl.map(function (im) { return im.subtract(mu).abs(); }).median().multiply(1.4826);
  var sigma = ee.Image.cat(IDX.map(function (k) { return sigRaw.select(k).max(FLOOR[k]).rename(k); }));
  var nBase = baseColl.count();
  var enough = baseColl.select('NDVI').count().gte(5);
  var withPrefix = function (img, p) { return img.select(IDX, IDX.map(function (k) { return p + k; })); };
  var stats = ee.Image.cat([withPrefix(mu, 'mu_'), withPrefix(sigma, 'sigma_'),
    withPrefix(sigRaw, 'sigraw_'), withPrefix(nBase, 'nbase_')]).updateMask(enough).toFloat();

  // ---------- sanity check: mean values over a 3 km radius test area at the oblast centre
  // Expected: sigma_NDVI ~ 0.03-0.08, sigma_LST ~ 1-3 deg C, nbase ~ 7.
  var test = ee.Geometry.Point([69.78, 51.55]).buffer(3000);
  print('Bands', stats.bandNames());
  print('Test region mean', stats.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: test, crs: P.crs, crsTransform: P.transform, maxPixels: 1e9
  }));

  Export.image.toAsset({
    image: stats, description: 'baseline_stats_2013_2019', assetId: STATS_ID,
    region: aoi, crs: P.crs, crsTransform: P.transform, maxPixels: 1e13
  });
  Map.centerObject(test, 11);
  Map.addLayer(stats.select('sigma_NDVI'), {min: 0.01, max: 0.12, palette: ['#f0efec', '#2a78d6', '#0d366b']}, 'sigma NDVI');
}

if (STATS_ASSET_READY) {
  // ---------- sample export: same stack as script 04 with baseline statistics appended
  var aBands = [];
  IDX.forEach(function (k) { YEARS.forEach(function (y) { aBands.push(k + '_' + y); }); });
  var mBands = ['severity', 'mechanism', 'T_NDVI', 'T_NDMI', 'T_LST', 'T_TGSI', 'T_SI',
    'NDVI_beta', 'NDMI_beta', 'LST_beta', 'TGSI_beta', 'SI_beta',
    'NDVI_tau', 'NDMI_tau', 'LST_tau', 'TGSI_tau', 'SI_tau'];
  var hBands = [];
  IDX.forEach(function (k) { HOLD.forEach(function (y) { hBands.push(k + '_' + y); }); });
  var tier = maskAsset.select('confidence').unmask(0).round().toInt().rename('tier');
  var district = ee.Image().toInt().paint(adm2, 'ADM2_CODE').unmask(0).rename('adm2_code');
  var lonlat = ee.Image.pixelLonLat().rename(['lon', 'lat']);
  var glcAll = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual').filterBounds(aoi).mosaic();
  var glcYears = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022];
  var glcNames = glcYears.map(function (y) { return 'GLC_' + y; });
  var glc = glcAll.select(glcYears.map(function (y) { return 'b' + (y - 1999); }), glcNames).unmask(0).toInt();

  var stack = A.select(aBands).addBands(M.select(mBands)).addBands(tier).addBands(district)
    .addBands(lonlat).addBands(glc)
    .addBands(ee.Image(ROOT + 'A_holdout_' + HOLD.join('_')).select(hBands))
    .addBands(ee.Image(STATS_ID).select(STAT_NAMES));
  var cols = aBands.concat(mBands, ['tier', 'adm2_code', 'lon', 'lat'], glcNames, hBands, STAT_NAMES);

  var samp = stack.sample({
    region: aoi, numPixels: SAMPLE.numPixels, seed: SAMPLE.seed,
    dropNulls: false, tileScale: SAMPLE.tileScale, geometries: false
  }).filter(ee.Filter.notNull(['severity']));

  Export.table.toDrive({
    collection: samp, description: 'mittcda_sample_v4_sigma',
    folder: FOLDER, fileFormat: 'CSV', selectors: cols
  });
  print('Table columns (expected 121)', cols.length);
  print('Baseline stats asset bands', ee.Image(STATS_ID).bandNames());
}
