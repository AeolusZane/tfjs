import * as tf from '@tensorflow/tfjs';
import { callbacks } from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

(async () => {
  //15%数据集昨验证集 xTrain训练集的输入特征 yTrain训练集的输出结果
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 10,
      //输出的特征长度
      inputShape: [xTrain.shape[1]],
      //[0-1之间的]激活函数
      activation: 'sigmoid',
    })
  );

  model.add(
    tf.layers.dense({
      units: IRIS_CLASSES.length,
      activation: 'softmax',
    })
  );

  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.1),
    //训练时要输出正确度 黄线是验证集 蓝线是训练集
    metrics: ['accuracy'],
  });

  await model.fit(xTrain, yTrain, {
    epochs: 100,
    validationData: [xTest, yTest],
    callbacks: tfvis.show.fitCallbacks(
      {
        name: '训练效果',
      },
      //损失函数 验证损失 准确度 验证准确度
      ['loss', 'val_loss', 'acc', 'val_acc'],
      { callbacks: ['onEpochEnd'] }
    ),
  });

  const input = tf.tensor2d([[5.1, 3.5, 1.4, 0.2]]);
  const pred = model.predict(input);
  //argMax返回最大值的索引
  console.log('预测结果', IRIS_CLASSES[pred.argMax(1).dataSync()[0]]);
})();
