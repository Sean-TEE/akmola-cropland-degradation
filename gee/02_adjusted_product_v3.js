/**** MITT-CDA · 02_adjusted_product_v3.js *************************************
 * MITT-CDA v3 full-raster product: applies the same decision rules as v2 after
 * subtracting the local year effect (median anomaly of surrounding cropland)
 * and using exact Kendall critical tau values for n = 8..12 valid years (pixels
 * with n < 8 are not flagged by the trend branch). Source assets are unchanged.
 *
 * Three stages controlled by STAGE:
 *   'A' — export reference tables of the year effect:
 *           grid   — median per CELL_M grid cell (default 25 km), using fineMedian().
 *   'B' — build and export the v3 asset for the chosen REF ('region' or 'grid').
 *   'C' — export v3 areas and district summaries (run after Stage B completes).
 *
 * Workflow: STAGE = 'A' -> wait for tasks -> STAGE = 'B' (REF = 'grid')
 *           -> wait for asset export -> STAGE = 'C'.
 *****************************************************************************/

var STAGE = 'A';            // 'A' | 'B' | 'C'
var REF = 'region';         // 'region' | 'grid' (for stages B and C)
var CELL_M = 25000;         // grid cell size in m (sensitivity: 50000)
var MIN_PIX = 10000;        // minimum cropland pixels per cell; otherwise use regional median
var GRID_CRS = 'EPSG:32642';
var ROOT = 'projects/akmola/assets/';
var FOLDER = 'akmola_masks';

var A = ee.Image(ROOT + 'A_stack_v2_2013_2024');
var M = ee.Image(ROOT + 'MITTCDA_v2_2013_2024');
var aoi = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Kazakhstan'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya')).geometry();

var IDX = ['NDVI', 'NDMI', 'LST', 'TGSI', 'SI'];
var YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var EVAL = [2020, 2021, 2022, 2023, 2024];
var bands = [];
IDX.forEach(function (k) { YEARS.forEach(function (y) { bands.push(k + '_' + y); }); });

var TAU_A = 1.5, GAMMA = 1.0, ALPHA = 0.6;
var W = {NDVI: 0.200, NDMI: 0.218, LST: 0.212, TGSI: 0.194, SI: 0.176};
// Kendall's critical tau, two-sided p < 0.05, exact null distribution; n = 8..12
var N_LIST = [8, 9, 10, 11, 12];
var TAU_CRIT = [0.6428, 0.5555, 0.5111, 0.4909, 0.4545];

var P = A.projection().getInfo();
var crop = M.select('severity').mask();

// fine-binned median: series clamped to +-CLAMP, histogram bin width ~0.001
var CLAMP = 8;
function fineMedian() {
  return ee.Reducer.median({maxBuckets: 4096, minBucketWidth: 0.004});
}
var ID_REGION = ROOT + 'ref_region_median_v3';
var ID_GRID = ROOT + 'ref_grid_median_' + (CELL_M / 1000) + 'km_v3';
var ID_V3 = ROOT + 'MITTCDA_v3_' + REF + (REF === 'grid' ? '_' + (CELL_M / 1000) + 'km' : '') + '_2013_2024_r2';

// ============================ STAGE A =====================================
if (STAGE === 'A') {
  var Am = A.select(bands).updateMask(crop).clamp(-CLAMP, CLAMP);

  // reducer check on a test region: fineMedian must match the exact median
  var testA = ee.Geometry.Point([69.78, 51.55]).buffer(5000);
  var rrA = function (img, reducer) {
    return img.reduceRegion({reducer: reducer, geometry: testA, crs: P.crs, crsTransform: P.transform, maxPixels: 1e9});
  };
  var valsA = ee.List(rrA(Am.select('LST_2023'), ee.Reducer.toList()).get('LST_2023')).sort();
  print('Test region, LST_2023: exact median', valsA.get(valsA.size().divide(2).floor().toInt()));
  print('  fineMedian (expected within 0.005 of exact)', rrA(Am.select('LST_2023'), fineMedian()));

  // one axis at a time (12 bands): 60 bands at once exceed memory limits
  var med = ee.Dictionary({});
  IDX.forEach(function (k) {
    var bk = YEARS.map(function (y) { return k + '_' + y; });
    med = med.combine(Am.select(bk).reduceRegion({
      reducer: fineMedian(), geometry: aoi,
      crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
    }));
  });
  // regional table asset export disabled: values are hardcoded in REGION_MED below

  var cells = aoi.coveringGrid(ee.Projection(GRID_CRS), CELL_M);
  print('Grid cells', cells.size());
  // one table per axis: five asset exports and five CSV copies to Drive
  IDX.forEach(function (k) {
    var bk = YEARS.map(function (y) { return k + '_' + y; });
    var gridStats = Am.select(bk).reduceRegions({
      collection: cells,
      reducer: fineMedian().combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
      crs: P.crs, crsTransform: P.transform, tileScale: 16
    });
    Export.table.toAsset({
      collection: gridStats, description: 'ref_grid_' + (CELL_M / 1000) + 'km_v3_' + k, assetId: ID_GRID + '_' + k
    });
    Export.table.toDrive({
      collection: gridStats, description: 'ref_grid_' + (CELL_M / 1000) + 'km_v3_' + k + '_csv',
      folder: FOLDER, fileFormat: 'CSV'
    });
  });
  Map.centerObject(aoi, 7);
  Map.addLayer(cells.style({color: '#52514e', fillColor: '00000000', width: 1}), {}, '25 km grid');
}

// ============================ Reference surface ==========================
// Regional median of A_k(t) on the full raster (from diag_regional_median_A_v3).
// Inline constants avoid exporting an asset that exceeds memory limits as a single table.
var REGION_MED = {
  NDVI_2013: 0.3624, NDVI_2014: 0.1539, NDVI_2015: -0.1220, NDVI_2016: 0.0004, NDVI_2017: 0.4100, NDVI_2018: -0.0584, NDVI_2019: -0.1169, NDVI_2020: 0.6380, NDVI_2021: 1.0540, NDVI_2022: 0.9140, NDVI_2023: 1.0140, NDVI_2024: -0.5380,
  NDMI_2013: 0.4222, NDMI_2014: 0.1203, NDMI_2015: -0.1343, NDMI_2016: 0.0004, NDMI_2017: 0.3100, NDMI_2018: -0.0217, NDMI_2019: -0.1204, NDMI_2020: 0.5220, NDMI_2021: 0.7020, NDMI_2022: 0.6420, NDMI_2023: 0.7220, NDMI_2024: -0.6220,
  LST_2013: 0.3339, LST_2014: 0.6745, LST_2015: -0.1381, LST_2016: -0.2820, LST_2017: 0.5022, LST_2018: -0.2980, LST_2019: -0.2462, LST_2020: 1.0020, LST_2021: 1.2700, LST_2022: 1.1700, LST_2023: 1.6300, LST_2024: -0.7540,
  TGSI_2013: 0.2460, TGSI_2014: 0.2581, TGSI_2015: -0.0700, TGSI_2016: 0.1222, TGSI_2017: 0.1140, TGSI_2018: 0.0001, TGSI_2019: -0.1742, TGSI_2020: 0.9580, TGSI_2021: 0.5620, TGSI_2022: 0.6820, TGSI_2023: 0.3220, TGSI_2024: 0.0060,
  SI_2013: -0.5540, SI_2014: 0.0000, SI_2015: 0.0300, SI_2016: -0.3380, SI_2017: 0.6180, SI_2018: -0.0460, SI_2019: 0.0000, SI_2020: -0.0180, SI_2021: 1.0380, SI_2022: 0.8100, SI_2023: 0.9460, SI_2024: -0.7460
};
function referenceImage() {
  var region = ee.Dictionary(REGION_MED).toImage(bands);
  if (REF === 'region') { return region; }
  // Cells are reduced to their centroids: a cell polygon touches eight neighbouring 25 km
  // grid pixels, and reduceToImage with first() could pick an arbitrary neighbour.
  // A centroid point falls inside exactly one grid pixel.
  var gridPts = {};
  IDX.forEach(function (k) {
    gridPts[k] = ee.FeatureCollection(ID_GRID + '_' + k)
      .filter(ee.Filter.gte(k + '_2020_count', MIN_PIX))
      .map(function (f) { return f.setGeometry(f.geometry().centroid(1)); });
  });
  var local = ee.Image.cat(bands.map(function (b) {
    var k = b.split('_')[0];
    return gridPts[k].reduceToImage([b + '_median'], ee.Reducer.first()).rename(b);
  })).reproject({crs: GRID_CRS, scale: CELL_M}).resample('bilinear');
  return local.unmask(region);          // fallback to regional median for cells with too few pixels
}

// ============================ MITT-CDA rules =============================
function activeAxis(img, k) {
  return ee.ImageCollection(EVAL.map(function (y) {
    return img.select([k + '_' + y], ['v']).gt(TAU_A);
  })).sum().gte(3).rename('I_' + k);
}
function trendAxis(img, k) {
  var coll = ee.ImageCollection(YEARS.map(function (y) {
    return ee.Image.constant(y).toFloat().rename('t').addBands(img.select([k + '_' + y], [k]));
  }));
  var slope = coll.reduce(ee.Reducer.sensSlope()).select('slope');
  var tau = coll.reduce(ee.Reducer.kendallsCorrelation(2)).select('tau');
  var n = coll.select(k).count();
  var crit = n.remap(N_LIST, TAU_CRIT, 9);          // threshold unreachable when n < 8
  var flag = slope.gt(0).and(tau.gte(crit));
  return ee.Image.cat([flag.rename('T_' + k), slope.rename(k + '_beta'), tau.rename(k + '_tau'), n.rename(k + '_n')]);
}
function phi(x) {
  return ee.Image(1).divide(ee.Image(1).add(x.subtract(TAU_A).multiply(-GAMMA).exp()));
}
function buildProduct(img) {
  var I = ee.Image.cat(IDX.map(function (k) { return activeAxis(img, k); }));
  var sev = I.reduce(ee.Reducer.sum()).rename('severity');
  var nd = I.select('I_NDVI'), nm = I.select('I_NDMI'), ls = I.select('I_LST'),
      tg = I.select('I_TGSI'), si = I.select('I_SI');
  // precedence D2 > D4 > D3 > D1, as in v2: 1 = D1, 2 = D3, 3 = D4, 4 = D2
  var mech = ee.Image(0).where(nd.and(ls), 1).where(nd.and(si), 2).where(nd.and(tg), 3)
    .where(nd.and(nm).and(ls), 4).rename('mechanism');
  var tr = IDX.map(function (k) { return trendAxis(img, k); });
  var T = ee.Image.cat(tr.map(function (t, i) { return t.select('T_' + IDX[i]); }));
  var betas = ee.Image.cat(tr.map(function (t, i) { return t.select(IDX[i] + '_beta'); }));
  var taus = ee.Image.cat(tr.map(function (t, i) { return t.select(IDX[i] + '_tau'); }));
  var ns = ee.Image.cat(tr.map(function (t, i) { return t.select(IDX[i] + '_n'); }));
  var DItrend = ee.Image.cat(IDX.map(function (k) { return T.select('T_' + k).multiply(W[k]); }))
    .reduce(ee.Reducer.sum()).rename('DI_trend');
  var DIthr = ee.ImageCollection(EVAL.map(function (y) {
    return ee.Image.cat(IDX.map(function (k) { return phi(img.select(k + '_' + y)).multiply(W[k]); }))
      .reduce(ee.Reducer.sum());
  })).mean().rename('DI_thr');
  var DI = DIthr.multiply(ALPHA).add(DItrend.multiply(1 - ALPHA)).rename('DI');
  return ee.Image.cat([DI, DIthr, DItrend, sev, mech, T, betas, taus, ns]).toFloat();
}

// ============================ STAGE B =====================================
if (STAGE === 'B') {
  var ref = referenceImage();
  // reference surface check: at each cell centre it must match the table median
  if (REF === 'grid') {
    var chkCells = ee.FeatureCollection(ID_GRID + '_NDVI').filter(ee.Filter.gte('NDVI_2020_count', 300000)).limit(4);
    print('Reference check: NDVI_2021 at cell centre | table median (expected within 0.01)',
      chkCells.map(function (f) {
        var v = ref.select('NDVI_2021').reduceRegion({
          reducer: ee.Reducer.first(), geometry: f.geometry().centroid(1), crs: GRID_CRS, scale: 100});
        return ee.Feature(null, {ref_at_centre: v.get('NDVI_2021'), table_median: f.get('NDVI_2021_median')});
      }));
  }
  var Aadj = A.select(bands).subtract(ref.select(bands)).rename(bands);
  var out = buildProduct(Aadj).updateMask(crop);
  print('v3 bands', out.bandNames());
  Export.image.toAsset({
    image: out, description: 'MITTCDA_v3_' + REF + '_r2', assetId: ID_V3,
    region: aoi, crs: P.crs, crsTransform: P.transform, maxPixels: 1e13
  });
  // rule check on a small test region: without subtraction, the output must match v2
  var test = ee.Geometry.Point([69.78, 51.55]).buffer(5000);
  var chk = buildProduct(A.select(bands));
  var same = chk.select(['mechanism', 'T_NDVI', 'T_SI']).eq(M.select(['mechanism', 'T_NDVI', 'T_SI'])).updateMask(crop)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: test, crs: P.crs, crsTransform: P.transform, maxPixels: 1e9});
  print('Check: agreement with v2 without subtraction (mechanism = 1; T_* = 1 except pixels with n < 12)', same);
  Map.centerObject(test, 11);
  Map.addLayer(ref.select('NDVI_2021'), {min: 0, max: 2, palette: ['#fcfcfb', '#eb6834']}, 'reference surface NDVI_2021');
  Map.addLayer(out.select('mechanism').selfMask(), {min: 1, max: 4, palette: ['#6250d6', '#eda100', '#8a4b00', '#e34948']}, 'v3: mechanism');
}

// ============================ STAGE C =====================================
if (STAGE === 'C') {
  var V3 = ee.Image(ID_V3);
  var pa = ee.Image.pixelArea().divide(1e6);
  var TB = ['T_NDVI', 'T_NDMI', 'T_LST', 'T_TGSI', 'T_SI'];
  var T3 = V3.select('mechanism').unmask(0).gt(0), R3 = V3.select(TB).unmask(0).reduce(ee.Reducer.max()).gt(0);
  var T2 = M.select('mechanism').unmask(0).gt(0), R2 = M.select(TB).unmask(0).reduce(ee.Reducer.max()).gt(0);
  var m3 = V3.select('mechanism').unmask(0);
  var layers = {
    area: ee.Image(1),
    T_v3: T3, R_v3: R3, TandR_v3: T3.and(R3), TorR_v3: T3.or(R3),
    T_v2: T2, R_v2: R2, TandR_v2: T2.and(R2),
    T_both: T3.and(T2), R_both: R3.and(R2),
    D1_v3: m3.eq(1), D3_v3: m3.eq(2), D4_v3: m3.eq(3), D2_v3: m3.eq(4),
    R_ge2_v3: V3.select(TB).unmask(0).reduce(ee.Reducer.sum()).gte(2)
  };
  TB.forEach(function (b) { layers[b + '_v3'] = V3.select(b).unmask(0).gt(0); });
  var names = Object.keys(layers);
  var stack = ee.Image.cat(names.map(function (nm) { return pa.multiply(layers[nm]).rename(nm + '_km2'); }))
    .updateMask(crop);
  var sums = stack.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi,
    crs: P.crs, crsTransform: P.transform, maxPixels: 1e13, tileScale: 16
  });
  Export.table.toDrive({
    collection: ee.FeatureCollection([ee.Feature(null, sums).set('ref', REF, 'cell_m', CELL_M)]),
    description: 'v3_areas_' + REF + '_r2', folder: FOLDER, fileFormat: 'CSV'
  });
  // areas by district (14 GAUL level-2 units)
  var adm2 = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Akmolinskaya'));
  Export.table.toDrive({
    collection: stack.select(['area_km2', 'T_v3_km2', 'R_v3_km2', 'TandR_v3_km2', 'T_v2_km2', 'R_v2_km2', 'TandR_v2_km2'])
      .reduceRegions({collection: adm2, reducer: ee.Reducer.sum(), crs: P.crs, crsTransform: P.transform, tileScale: 16})
      .select(['ADM2_NAME', 'area_km2', 'T_v3_km2', 'R_v3_km2', 'TandR_v3_km2', 'T_v2_km2', 'R_v2_km2', 'TandR_v2_km2'], null, false),
    description: 'v3_areas_by_district_' + REF + '_r2', folder: FOLDER, fileFormat: 'CSV'
  });
}
