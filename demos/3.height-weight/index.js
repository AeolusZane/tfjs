import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

(async () => {
  const heights = [150, 160, 170];
  const weights = [40, 50, 60];
  tfvis.render.scatterplot(
    {
      name: '身高体重训练数据',
    },
    {
      values: heights.map((x, i) => ({ x, y: weights[i] })),
    },
    {
      xAxisDomain: [140, 180],
      yAxisDomain: [30, 70],
    }
  );
  //归一化操作不需要手动完成，tfjs会自动完成
  const inputs = tf.tensor(heights).sub(150).div(20);
  const labels = tf.tensor(weights).sub(40).div(20);
  //初始化1个连续的模型 上一层的输出是下一层的输入
  const model = tf.sequential();
  //构建1个 单层单个神经元的神经网络 添加到model
  //dense 是一个全连接层 能够对 output = activation(dot(input, kernel) + bias)
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  //准备训练模型
  model.compile({
    //均方误差损失函数 对计算的结果进行平方运算 求和 再求平均
    loss: tf.losses.meanSquaredError,
    //优化器 随机梯度下降 0.1是一个学习速率
    optimizer: tf.train.sgd(0.1),
  });
  //进行训练进行拟合
  await model.fit(inputs, labels, {
    batchSize: 3,
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks({ name: '训练模型' }, ['loss']),
  });

  //反归一化操作
  const output = model.predict(tf.tensor([180]).sub(150).div(20));
  console.log('180大哥的体重应该是', output.mul(20).add(40).dataSync()[0]);
})();
