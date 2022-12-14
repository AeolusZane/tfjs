import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

(async () => {
  const data = getData(400);
  //   console.log('data: ', data);
  tfvis.render.scatterplot(
    { name: '逻辑回归训练数据' },
    {
      values: [
        data.filter((p) => p.label === 1),
        data.filter((p) => p.label === 0),
      ],
    }
  );
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [2],
      //想得到一个概率值，所以使用sigmoid激活函数 [0-1]
      activation: 'sigmoid',
    })
  );
  model.compile({
    //对数损失函数 对输入的值进行log运算
    loss: tf.losses.logLoss,
    //算法优化器 自适应矩阵变化
    optimizer: tf.train.adam(0.1),
  });

  const inputs = tf.tensor(data.map((p) => [p.x, p.y]));
  const labels = tf.tensor(data.map((p) => p.label));

  await model.fit(inputs, labels, {
    //训练多少轮
    batchSize: 40,
    //每轮训练多少次
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks({ name: '训练效果' }, ['loss']),
  });

  const predata = [-0.2, 2];
  const pred = model.predict(tf.tensor([predata]));
  console.log('预测结果', pred.dataSync()[0]);
})();
