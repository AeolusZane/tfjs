import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

(async () => {
  const data = getData(400);
  console.log('data: ', data);
  const model = tf.sequential();
  tfvis.render.scatterplot(
    { name: 'XOR训练数据' },
    {
      values: [
        data.filter((p) => p.label === 1),
        data.filter((p) => p.label === 0),
      ],
    }
  );
  //hidden layer
  model.add(tf.layers.dense({ units: 4, inputShape: [2], activation: 'relu' }));
  //输出层
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    //交叉熵损失函数
    loss: tf.losses.logLoss,
    //自动调整学习率
    optimizer: tf.train.adam(0.1),
  });
  const inputs = tf.tensor(data.map((p) => [p.x, p.y]));
  const labels = tf.tensor(data.map((p) => p.label));

  await model.fit(inputs, labels, {
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({ name: '训练效果' }, ['loss']),
  });

  const predata = [4, 5];
  const pred = model.predict(tf.tensor([predata]));
  console.log('预测结果: ', pred.dataSync()[0]);
})();
