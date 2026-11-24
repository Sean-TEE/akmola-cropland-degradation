/**** MITT-CDA · 03_holdout_2025_2026.js ***************************************
 * Hold-out years (2025 and 2026): precipitation-adjusted anomalies computed
 * with the same Landsat 8 pipeline and 2013-2019 baseline as A_stack_v2.
 * These years play no part in deriving the T and R layers.
 *
 * What it does:
 *   1. Re-implements the Landsat 8 and ERA5-Land anomaly pipeline in JS.
 *   2. Checks reproducibility against A_stack_v2 for 2024 at five control points.
 *   3. Prints Landsat 8 scene counts for 2025 and 2026 and latest ERA5-Land date.
 *   4. Exports asset A_holdout_2025_2026 (10 bands, native grid) when RUN_EXPORTS = true.
 *   5. Exports sigma_floor_area and composite_doy_by_year CSVs when RUN_EXPORTS = true,
 *      and holdout_by_group CSV when HOLD_ASSET_READY = true.
 *****************************************************************************/

var HOLD = [2025, 2026];
var RUN_EXPORTS = false;       // set true to re-run asset, sigma-floor, and DOY exports
var ROOT = 'projects/akmola/assets/';
var FOLDER = 'akmola_masks';

var A = ee.Image(ROOT + 'A_stack_v2_2013_2024');
var M = ee.Image(ROOT + 'MITTCDA_v2_2013_2024');
var crop = ee.Image(ROOT + 'cropland_mask_v1').select('cropland');
var aoi = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Kazakhstan'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya')).geometry();
var P = A.projection().getInfo();
var cropA = M.select('severity').mask();          // published analysis area

var IDX = ['NDVI', 'NDMI', 'LST', 'TGSI', 'SI'];
var BASE = [2013, 2014, 2015, 2016, 2017, 2018, 2019];
var ALLY = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024].concat(HOLD);
var DELTA = {NDVI: -1, NDMI: -1, LST: 1, TGSI: 1, SI: 1};
var FLOOR = {NDVI: 0.01, NDMI: 0.01, TGSI: 0.005, SI: 0.002, LST: 0.3};
var B = {NDVI: 0.173, NDMI: 0.178, LST: -0.370, TGSI: -0.211, SI: 0.0};

// ---------- Landsat 8 C2 L2: cloud mask and annual median composite
function maskL8(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 2).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0)).and(qa.bitwiseAnd(1 << 4).eq(0));
  var optical = img.select('SR_B.').multiply(2.75e-05).add(-0.2);
  var thermal = img.select('ST_B10').multiply(0.00341802).add(149.0);
  var valid = optical.gt(0).and(optical.lt(1)).reduce(ee.Reducer.min());
  return img.addBands(optical, null, true).addBands(thermal, null, true)
    .updateMask(clear).updateMask(valid);
}
function scenes(year) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(year, 6, 15), ee.Date.fromYMD(year, 8, 15))
    .filter(ee.Filter.lt('CLOUD_COVER', 60));
}
function indicesForYear(year) {
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
}

// ---------- baseline and standardization
var baseColl = ee.ImageCollection(BASE.map(indicesForYear)).select(IDX);
var mu = baseColl.median();
var sigRaw = baseColl.map(function (im) { return im.subtract(mu).abs(); }).median().multiply(1.4826);
var sigma = ee.Image.cat(IDX.map(function (k) { return sigRaw.select(k).max(FLOOR[k]).rename(k); }));
var enough = baseColl.select('NDVI').count().gte(5);

// ---------- ERA5-Land precipitation over the same compositing window
function precip(year) {
  return ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(ee.Date.fromYMD(year, 6, 15), ee.Date.fromYMD(year, 8, 15))
    .select('total_precipitation_sum').sum().multiply(1000).rename('P');
}
var Pbase = ee.ImageCollection(BASE.map(precip));
var muP = Pbase.median();
var sigP = Pbase.map(function (im) { return im.subtract(muP).abs(); }).median().multiply(1.4826);
sigP = sigP.where(sigP.lt(1e-6), 1e-6);
function zP(year) { return precip(year).subtract(muP).divide(sigP).rename('ZP'); }

// ---------- precipitation-adjusted anomaly A_k(year)
function anomaly(year) {
  var z = indicesForYear(year).select(IDX).subtract(mu).divide(sigma).updateMask(enough);
  var zp = zP(year);
  return ee.Image.cat(IDX.map(function (k) {
    return z.select(k).subtract(zp.select('ZP').multiply(B[k])).multiply(DELTA[k]).rename(k + '_' + year);
  }));
}

// ---------- 2. reproducibility check at five control points
// Compares re-computed LST_2024 (largest |b|) against A_stack_v2: prints max, mean,
// and share of pixels differing by > 0.0001.
var test = ee.Geometry.Point([69.78, 51.55]).buffer(5000);
var pts = {west: [66.3, 51.6], north: [69.0, 52.5], centre: [69.78, 51.55], south: [70.9, 51.0], east: [72.6, 51.3]};
var dLst = anomaly(2024).select('LST_2024').subtract(A.select('LST_2024')).abs();
Object.keys(pts).forEach(function (nm) {
  var g = ee.Geometry.Point(pts[nm]).buffer(3000);
  print('LST_2024, ' + nm + ': max | mean | share > 0.0001',
    dLst.rename('d').addBands(dLst.gt(1e-4).rename('share')).reduceRegion({
      reducer: ee.Reducer.max().combine({reducer2: ee.Reducer.mean(), sharedInputs: true}),
      geometry: g, crs: P.crs, crsTransform: P.transform, maxPixels: 1e9
    }));
});

// ---------- 3. scene and ERA5-Land availability
HOLD.forEach(function (y) { print('Landsat 8 scenes in ' + y + ' (15 Jun - 15 Aug, cloud cover < 60%)', scenes(y).size()); });
print('Latest ERA5-Land date in catalog',
  ee.Date(ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').limit(1, 'system:time_start', false).first().get('system:time_start')));

// ---------- 4. hold-out asset
var hold = ee.Image.cat(HOLD.map(anomaly)).toFloat();
if (RUN_EXPORTS) Export.image.toAsset({
  image: hold, description: 'A_holdout_' + HOLD.join('_'), assetId: ROOT + 'A_holdout_' + HOLD.join('_'),
  region: aoi, crs: P.crs, crsTransform: P.transform, maxPixels: 1e13
});

// ---------- 5. summary tables
// 5a. Set HOLD_ASSET_READY = true after the hold-out asset export finishes to compute
//     hold-out median anomaly and exceedance (> 1.5) by v2 flag group:
//     grp: 0 = unflagged, 1 = T only, 2 = R only, 3 = T and R.
//     Only pixels with all ten hold-out bands unmasked are included.
var HOLD_ASSET_READY = false;
var holdBands = [];
IDX.forEach(function (k) { HOLD.forEach(function (y) { holdBands.push(k + '_' + y); }); });
if (HOLD_ASSET_READY) {
  var H = ee.Image(ROOT + 'A_holdout_' + HOLD.join('_')).select(holdBands);
  var fineMedian = function () {
    return ee.Reducer.median({maxBuckets: 4096, minBucketWidth: 0.004});
  };
  var TB = ['T_NDVI', 'T_NDMI', 'T_LST', 'T_TGSI', 'T_SI'];
  var T2 = M.select('mechanism').unmask(0).gt(0);
  var R2 = M.select(TB).unmask(0).reduce(ee.Reducer.max()).gt(0);
  var grp = T2.add(R2.multiply(2)).rename('grp');
  var nb = holdBands.length;
  var byGroup = function (img, reducer, key, stat) {
    var g = img.addBands(grp).updateMask(cropA).reduceRegion({
      reducer: reducer.repeat(nb).group({groupField: nb, groupName: 'grp'}), geometry: aoi,
      crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
    });
    return ee.FeatureCollection(ee.List(g.get('groups')).map(function (d) {
      d = ee.Dictionary(d);
      return ee.Feature(null, ee.Dictionary.fromLists(holdBands, ee.List(d.get(key)))).set('grp', d.get('grp'), 'stat', stat);
    }));
  };
  Export.table.toDrive({
    collection: byGroup(H.clamp(-8, 8), fineMedian(), 'median', 'median_A')
      .merge(byGroup(H.gt(1.5), ee.Reducer.mean(), 'mean', 'share_above_1.5')),
    description: 'holdout_by_group', folder: FOLDER, fileFormat: 'CSV',
    selectors: ['stat', 'grp'].concat(holdBands)
  });
}

// share of cropland area where sigma hits the lower floor
var pa = ee.Image.pixelArea().divide(1e6);
var atFloor = ee.Image.cat(IDX.map(function (k) {
  return pa.multiply(sigRaw.select(k).lte(FLOOR[k])).rename(k + '_floor_km2');
})).addBands(pa.rename('area_km2')).updateMask(enough).updateMask(cropA);
if (RUN_EXPORTS) Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(null, atFloor.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
  }))]),
  description: 'sigma_floor_area', folder: FOLDER, fileFormat: 'CSV'
});

// mean day-of-year (DOY) of clear composite observations and mean observation count
var doyRows = ALLY.map(function (y) {
  var coll = scenes(y).map(maskL8).map(function (img) {
    var doy = img.date().getRelative('day', 'year').add(1);
    return ee.Image.constant(doy).toFloat().rename('doy').updateMask(img.select('SR_B4').mask());
  });
  var st = coll.mean().rename('doy_mean').addBands(coll.count().rename('n_obs')).updateMask(cropA)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi, scale: 300, maxPixels: 1e10, tileScale: 8});
  return ee.Feature(null, st).set('year', y, 'scenes', scenes(y).size());
});
if (RUN_EXPORTS) Export.table.toDrive({
  collection: ee.FeatureCollection(doyRows), description: 'composite_doy_by_year',
  folder: FOLDER, fileFormat: 'CSV', selectors: ['year', 'scenes', 'doy_mean', 'n_obs']
});

Map.centerObject(test, 11);
Map.addLayer(hold.select('NDVI_' + HOLD[0]), {min: -2, max: 3, palette: ['#2a78d6', '#f0efec', '#e34948']}, 'A_NDVI ' + HOLD[0]);
Map.addLayer(M.select('mechanism').selfMask(), {min: 1, max: 4, palette: ['#6250d6', '#eda100', '#8a4b00', '#e34948']}, 'v2: mechanism', false);
