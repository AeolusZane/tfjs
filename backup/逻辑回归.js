const data = getData(400);
// console.log('data: ', data);
// tfvis.render.scatterplot(
//   {
//     name: '逻辑回归训练数据',
//   },
//   {
//     values: [
//       data.filter((p) => p.label === 1),
//       data.filter((p) => p.label === 0),
//     ],
//   }
// );
//归一化操作不需要手动完成，tfjs会自动完成
const inputs = tf.tensor(data, [data.length, 2]);
const labels = tf.tensor(data.map((p) => p.label));
//初始化1个连续的模型 上一层的输出是下一层的输入
const model = tf.sequential();
//构建1个 单层单个神经元的神经网络 添加到model
//dense 是一个全连接层 能够对 output = activation(dot(input, kernel) + bias)
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
//准备训练模型
model.compile({
  //均方误差损失函数 对计算的结果进行平方运算 求和 再求平均
  loss: tf.losses.sigmoidCrossEntropy,
  //优化器 随机梯度下降 0.1是一个学习速率
  optimizer: tf.train.adam(0.1),
});
//进行训练进行拟合
await model.fit(inputs, labels, {
  batchSize: 40,
  epochs: 20,
  callbacks: tfvis.show.fitCallbacks({ name: '训练模型' }, ['loss']),
});

window.predict = (form) => {
  const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
  alert(`预测结果：${pred.dataSync()[0]}`);
};
