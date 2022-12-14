export function getData(numSamples) {
  let points = [];

  function genGauss(cx, cy, label) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx);
      let y = normalRandom(cy);
      points.push({ x, y, label });
    }
  }
  //正态分布 + 高斯分布
  genGauss(2, 2, 1);
  genGauss(-2, -2, 0);
  return points;
}
//大数据分析的数据源 20-30 癌症的概率 大小
//box-muller transform算法 生成正态分布的随机数
function normalRandom(mean = 0, variance = 1) {
  let v1, v2, s;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt((-2 * Math.log(s)) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}
const data = getData(20);
// console.log('data: ', data);
