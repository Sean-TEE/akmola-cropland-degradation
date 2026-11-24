/**** MITT-CDA · 01_diagnostics.js ***********************************************
 * Diagnostics for the existing 2013-2024 anomaly stack (no new raster assets).
 * Uses fineMedian() (histogram with maxBuckets: 4096, minBucketWidth: 0.004)
 * because the default ee.Reducer.median() produces quantized regional medians
 * on large pixel counts.
 *
 * Steps:
 *   1. Prints the native grid projection and pixel area (EPSG:4326, ~560 m2).
 *   2. Verifies fineMedian() against an exact sorted median on a test region.
 *   3. Exports the regional median of A_k(t) on the native grid (60 values).
 *   4. Optionally exports valid-year summary tables when EXPORT_N_TABLES = true.
 *
 * Run -> verify fineMedian() matches the exact median in Console -> start task
 * diag_regional_median_A_v3.
 *****************************************************************************/

var EXPORT_N_TABLES = false;

var A = ee.Image('projects/akmola/assets/A_stack_v2_2013_2024');
var M = ee.Image('projects/akmola/assets/MITTCDA_v2_2013_2024');
var aoi = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Kazakhstan'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya')).geometry();

var FOLDER = 'akmola_masks';
var IDX = ['NDVI', 'NDMI', 'LST', 'TGSI', 'SI'];
var YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var bands = [];
IDX.forEach(function (k) { YEARS.forEach(function (y) { bands.push(k + '_' + y); }); });

// ---------- fine-binned median
// Values are clamped to +-CLAMP (does not affect the median); histogram bin width is ~0.001.
var CLAMP = 8;
function fineMedian() {
  return ee.Reducer.median({maxBuckets: 4096, minBucketWidth: 0.004});
}

// ---------- 1. asset grid
var P = A.projection().getInfo();                 // {crs: ..., transform: [...]}
print('A: crs and transform', P);
print('Native grid pixel area at oblast centre (m2)',
  ee.Image.pixelArea().reduceRegion({
    reducer: ee.Reducer.first(), geometry: aoi.centroid(1),
    crs: P.crs, crsTransform: P.transform
  }));

// analysis mask = published analysis domain (57,663 km2)
var crop = M.select('severity').mask();
var pa = ee.Image.pixelArea().divide(1e6);

// ---------- 2. reducer check: single band, test region, three methods
var test = ee.Geometry.Point([69.78, 51.55]).buffer(5000);
var tb = A.select('NDVI_2021').updateMask(crop);
function onTest(img, reducer) {
  return img.reduceRegion({reducer: reducer, geometry: test, crs: P.crs, crsTransform: P.transform, maxPixels: 1e9});
}
var vals = ee.List(onTest(tb, ee.Reducer.toList()).get('NDVI_2021')).sort();
print('Test region, NDVI_2021: pixel count', vals.size());
print('  exact median (sorted)', vals.get(vals.size().divide(2).floor().toInt()));
print('  fineMedian (expected within 0.005 of exact)', onTest(tb.clamp(-CLAMP, CLAMP), fineMedian()));
print('  default ee.Reducer.median() for comparison', onTest(tb, ee.Reducer.median()));

// ---------- 3. regional median on the native grid
// one axis at a time (12 bands): 60 bands at once exceed memory limits
var med = ee.Dictionary({});
IDX.forEach(function (k) {
  var bk = YEARS.map(function (y) { return k + '_' + y; });
  med = med.combine(A.select(bk).updateMask(crop).clamp(-CLAMP, CLAMP).reduceRegion({
    reducer: fineMedian(), geometry: aoi,
    crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
  }));
});
Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(null, med)]),
  description: 'diag_regional_median_A_v3', folder: FOLDER, fileFormat: 'CSV', selectors: bands
});

// ---------- 4-5. valid year counts (exported when EXPORT_N_TABLES is true)
function nValid(k) {
  return A.select(YEARS.map(function (y) { return k + '_' + y; }))
    .mask().reduce(ee.Reducer.sum()).round().toInt().rename('n_' + k);
}
var nMin = ee.Image.cat(IDX.map(nValid)).reduce(ee.Reducer.min()).rename('n_min');
if (EXPORT_N_TABLES) {
  var nByAxis = IDX.map(function (k) {
    var g = pa.addBands(nValid(k)).updateMask(crop).reduceRegion({
      reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'n'}), geometry: aoi,
      crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
    });
    return ee.FeatureCollection(ee.List(g.get('groups')).map(function (d) {
      d = ee.Dictionary(d);
      return ee.Feature(null, {axis: k, n_valid: d.get('n'), km2: d.get('sum')});
    }));
  });
  Export.table.toDrive({
    collection: ee.FeatureCollection(nByAxis).flatten(),
    description: 'diag_n_valid_years_by_axis', folder: FOLDER, fileFormat: 'CSV',
    selectors: ['axis', 'n_valid', 'km2']
  });
  var R2 = M.select(['T_NDVI', 'T_NDMI', 'T_LST', 'T_TGSI', 'T_SI']).unmask(0).reduce(ee.Reducer.max()).gt(0);
  var T2 = M.select('mechanism').unmask(0).gt(0);
  var st = pa.rename('km2')
    .addBands(pa.multiply(T2).rename('T_km2'))
    .addBands(pa.multiply(R2).rename('R_km2'))
    .addBands(pa.multiply(T2.and(R2)).rename('TR_km2'))
    .addBands(nMin).updateMask(crop);
  var gT = st.reduceRegion({
    reducer: ee.Reducer.sum().repeat(4).group({groupField: 4, groupName: 'n_min'}), geometry: aoi,
    crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
  });
  Export.table.toDrive({
    collection: ee.FeatureCollection(ee.List(gT.get('groups')).map(function (d) {
      d = ee.Dictionary(d);
      var s = ee.List(d.get('sum'));
      return ee.Feature(null, {n_min: d.get('n_min'), km2: s.get(0), T_km2: s.get(1), R_km2: s.get(2), TR_km2: s.get(3)});
    })),
    description: 'diag_T_R_by_n_min', folder: FOLDER, fileFormat: 'CSV',
    selectors: ['n_min', 'km2', 'T_km2', 'R_km2', 'TR_km2']
  });
}

Map.centerObject(aoi, 7);
Map.addLayer(nMin.updateMask(crop), {min: 5, max: 12, palette: ['#d03b3b', '#fab219', '#cde2fb', '#2a78d6']}, 'n_min: valid years');
