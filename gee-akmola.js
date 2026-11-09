var di = ee.Image('projects/akmola/assets/MITTCDA_v2_2013_2024');
print('Bands:', di.bandNames());
var adm = ee.FeatureCollection('FAO/GAUL/2015/level2')
            .filter(ee.Filter.eq('ADM1_NAME','Aqmola'));
print('Districts:', adm.aggregate_array('ADM2_NAME'));
Map.addLayer(adm, {}, 'adm');


print(di.select(['DI_thr','DI_trend']).reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: di.geometry(),
  scale: 3000, maxPixels: 1e10, bestEffort: true
}));

var g2 = ee.FeatureCollection('FAO/GAUL/2015/level2')
           .filterBounds(di.geometry());
print('ADM1 в охвате растра:', g2.aggregate_array('ADM1_NAME').distinct());
print('features:', g2.size());

ee.FeatureCollection('projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM2')
  .filter(ee.Filter.eq('shapeGroup','KAZ'))
  
  
  
var di   = ee.Image('projects/akmola/assets/MITTCDA_v2_2013_2024');
var mask = ee.Image('projects/akmola/assets/cropland_mask_v1');
var crop = mask.gt(0);

var frac = ee.Image.cat([
  di.select('DI_thr').gt(0).rename('f_thr'),
  di.select('DI_trend').gt(0).rename('f_trd'),
  di.select('DI_thr').gt(0).and(di.select('DI_trend').gt(0)).rename('f_both')
]).updateMask(crop);

print('Доли при правиле gt(0):', frac.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: di.geometry(),
  scale: 300,
  maxPixels: 1e10
}));


print('severity/mechanism:', di.select(['severity','mechanism']).reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: di.geometry(),
  scale: 1000, maxPixels: 1e10, bestEffort: true
}));

var adm = ee.FeatureCollection('FAO/GAUL/2015/level2')
            .filter(ee.Filter.eq('ADM1_NAME','Akmolinskaya'));
print('n =', adm.size());
print('ADM2:', adm.aggregate_array('ADM2_NAME'));
Map.addLayer(adm, {}, 'Akmolinskaya ADM2');



var sko = ee.FeatureCollection('FAO/GAUL/2015/level2')
            .filter(ee.Filter.eq('ADM1_NAME','Severo-kazachstanskaya'));
print('SKO ADM2:', sko.aggregate_array('ADM2_NAME'));

var extra = sko.filter(ee.Filter.inList('ADM2_NAME',
  ['Zerendinskiy','Enbekshilderskiy','Shchuchinskiy']));  // сверь написание!
var adm17 = adm.merge(extra);
print('n =', adm17.size());   // ждёшь 17
Map.addLayer(adm17, {color:'red'}, 'adm17');

var mask = ee.Image('projects/akmola/assets/cropland_mask_v1');
print('Mask bands:', mask.bandNames());
var pa = ee.Image.pixelArea().divide(1e6);
print('Площадь каждого бэнда, км²:', mask.gt(0).multiply(pa).reduceRegion({
  reducer: ee.Reducer.sum(), geometry: mask.geometry(),
  scale: 300, maxPixels: 1e10
}));


var extra = sko.filter(ee.Filter.inList('ADM2_NAME',
  ['Zerendy','Engbekshilder','Shuchinskiy']));
var adm17 = adm.merge(extra);
print('n =', adm17.size());   // обязано стать 17
Map.addLayer(adm17, {color:'red'}, 'adm17');

var crop = ee.Image('projects/akmola/assets/cropland_mask_v1')
             .select('cropland').gt(0);
             
var di = ee.Image('projects/akmola/assets/MITTCDA_v2_2013_2024');
var frac = ee.Image.cat([
  di.select('DI_thr').gt(0).rename('f_thr'),
  di.select('DI_trend').gt(0).rename('f_trd'),
  di.select('DI_thr').gt(0).and(di.select('DI_trend').gt(0)).rename('f_both')
]).updateMask(crop);

print('Доли:', frac.reduceRegion({
  reducer: ee.Reducer.mean(), geometry: di.geometry(),
  scale: 300, maxPixels: 1e10
}));

var pa = ee.Image.pixelArea();
var stack = ee.Image.cat([
  di.select('DI_thr').gt(0).multiply(pa).rename('a_thr'),
  di.select('DI_trend').gt(0).multiply(pa).rename('a_trd'),
  di.select('DI_thr').gt(0).and(di.select('DI_trend').gt(0)).multiply(pa).rename('a_both'),
  pa.rename('a_crop')
]).updateMask(crop);

Export.table.toDrive({
  collection: stack.reduceRegions({
    collection: adm17, reducer: ee.Reducer.sum(), scale: 30, tileScale: 4
  }).select(['ADM2_NAME','a_thr','a_trd','a_both','a_crop'], null, false),
  description: 'MITTCDA_by_district', fileFormat: 'CSV'
});

// NAME_MAP = {
// 'Akkol':'Аккольский','Arshaly':'Аршалынский','Astrakhanskiy':'Астраханский',
// 'Atbasar':'Атбасарский','Bulandy':'Буландынский','Egindykol':'Егиндыкольский',
// 'Ereymentau':'Ерейментауский','Esil':'Есильский','Korghalzhyn':'Коргалжынский',
// 'Sandyktau':'Сандыктауский','Shortandy':'Шортандинский','Tselinograd':'Целиноградский',
// 'Zhaksy':'Жаксынский','Zharkaiyn':'Жаркаинский','Zerendy':'Зерендинский',
// 'Engbekshilder':'Биржан сал','Shuchinskiy':'Бурабайский'}
 
 
 
 // бинарны ли пороговые флаги T_*?
print('T_* histogram:', di.select(['T_NDVI','T_NDMI','T_LST','T_TGSI','T_SI'])
  .reduceRegion({reducer: ee.Reducer.frequencyHistogram(),
    geometry: di.geometry(), scale: 3000, maxPixels: 1e10, bestEffort: true}));

var tsum = di.select('T_NDVI').add(di.select('T_NDMI')).add(di.select('T_LST'))
             .add(di.select('T_TGSI')).add(di.select('T_SI'));

print('Кандидаты:', ee.Image.cat([
  di.select('mechanism').gt(0).rename('mech_gt0'),
  di.select('severity').gt(0).rename('sev_gt0'),
  tsum.gte(1).rename('T_ge1'),
  tsum.gte(2).rename('T_ge2')
]).updateMask(crop).reduceRegion({
  reducer: ee.Reducer.mean(), geometry: di.geometry(),
  scale: 300, maxPixels: 1e10
}));