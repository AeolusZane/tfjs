import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
// import { getData } from '../5.xor/data';
import { getData } from './data';
(async () => {
  //本来是1分为二的一刀切的问题 但是算出了曲线就不对了 过拟合
  //简单模型解决复杂了 训练时间不够
  const data = getData(200, 2);
  tfvis.render.scatterplot(
    { name: '训练数据' },
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
      units: 10,
      inputShape: [2],
      activation: 'tanh',
      //设置l2正则化 使得权重不要太大 等于设置了权重衰减
      //   kernelRegularizer: tf.regularizers.l2({ l2: 1 }),
    })
  );
  model.add(tf.layers.dropout({ rate: 0.05 }));
  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
    })
  );
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1),
  });
  const inputs = tf.tensor(data.map((p) => [p.x, p.y]));
  const labels = tf.tensor(data.map((p) => p.label));

  await model.fit(inputs, labels, {
    //从训练集中分出来一部分验证集
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: '训练效果',
      },
      ['loss', 'val_loss'],
      { callbacks: ['onEpochEnd'] }
    ),
  });
})();
