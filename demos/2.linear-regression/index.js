import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

(async () => {
  //特征值 ===输入
  const xs = [1, 2, 3, 4];
  //标签值 ===输出
  const ys = [1, 3, 5, 7];
  //散点图
  tfvis.render.scatterplot(
    {
      name: '第一个线性回归的训练集',
    },
    {
      values: xs.map((x, i) => ({ x, y: ys[i] })),
    },
    {
      xAxisDomain: [0, 5],
      yAxisDomain: [0, 8],
    }
  );
  /**
   * 1.构建单层单个的神经元 神经网络
   * 2.dense全链接层output = activation(dot(input, kernel) + bias)
   * 3.units是神经元的个数 也可以直接做决断不需要激活函数
   * https://js.tensorflow.org/api/latest/#layers.dense
   */

  //创建1个连续的模型 下一层的输入一定是上一层的输出
  const model = tf.sequential();
  //添加一个全连接层
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  /*1.准备训练模型
   * https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss?hl=zh-cn
   * 2.第一次的损失函数的 他是瞎蒙的 借助优化器不行的去调整
   * 3.SGD 随机梯度下降
   */
  model.compile({
    //损失函数：均方误差损失函数
    loss: tf.losses.meanSquaredError,
    //梯度下降优化算法
    //0.1不是固定的  学习速率
    optimizer: tf.train.sgd(0.1),
  });
  //可视化过程 + 正式进行训练流程
  const inputs = tf.tensor(xs);
  const labels = tf.tensor(ys);
  await model.fit(inputs, labels, {
    //随机梯度下降的小批量参数
    batchSize: 1,
    epochs: 100,
    //从损失函数上进行优化的可视化过程
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
  });

  const output = model.predict(tf.tensor([5]));
  //   console.log('output: ', output);
  console.log(`x为5预测的值${Array.from(output.dataSync())}`);
})();
