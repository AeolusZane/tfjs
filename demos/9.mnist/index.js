import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

(async () => {
  console.log('Loading data...');
  const data = new MnistData();
  await data.load();
  //加载验证集
  const examples = data.nextTestBatch(20);

  //放置图片的区域
  const surface = tfvis.visor().surface({ name: '输入示例' });

  for (let i = 0; i < 20; i += 1) {
    const imageTensor = tf.tidy(() => {
      //   return data.nextTestBatch(20).xs.reshape([28, 28, 1]);
      return examples.xs.slice([i, 0], [1, 784]).reshape([28, 28, 1]);
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);
    // imageTensor.dispose();
  }
  //卷积层 池化层 输出层
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      //卷积核
      kernelSize: 5,
      //卷积核数量
      filters: 8,
      //移动步长
      strides: 1,
      //激活函数
      activation: 'relu',
      //输出形状
      kernelInitializer: 'varianceScaling',
    })
  );

  model.add(
    tf.layers.maxPooling2d({
      //池化尺寸
      poolSize: [2, 2],
      //移动步长
      strides: [2, 2],
    })
  );
  model.add(
    //卷积层2维图片
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      //卷积核
      kernelSize: 5,
      //卷积核数量
      filters: 16,
      //移动步长
      strides: 1,
      //激活函数
      activation: 'relu',
      //输出形状
      kernelInitializer: 'varianceScaling',
    })
  );
  model.add(
    tf.layers.maxPool2d({
      //池化尺寸
      poolSize: [2, 2],
      //移动步长
      strides: [2, 2],
    })
  );

  //高维数据放在最后一层前展开数据层
  model.add(tf.layers.flatten());

  model.add(
    //输出层的输出形状得和上一步保持一致
    tf.layers.dense({
      units: 10,
      // [0.1,0.2,0.3 = 1] 激活函数
      activation: 'softmax',
      kernelInitializer: 'varianceScaling',
    })
  );

  model.compile({
    //损失函数 交叉熵损失函数 输出0-1之间的概率值
    loss: 'categoricalCrossentropy',
    //自动调整学习率 优化函数
    optimizer: tf.train.adam(),
    //度量单位的准确度
    metrics: ['accuracy'],
  });

  //训练模型
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(1000);
    return [d.xs.reshape([1000, 28, 28, 1]), d.labels];
  });
  //验证数据
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(200);
    return [d.xs.reshape([200, 28, 28, 1]), d.labels];
  });

  //训练模型

  await model.fit(trainXs, trainYs, {
    validationData: [testXs, testYs],
    batchSize: 500,
    epochs: 20,
    callbacks: [
      tfvis.show.fitCallbacks({ name: '训练效果' }, [
        'loss',
        'val_loss',
        'acc',
        'val_acc',
      ]),
      { callbacks: ['onEpochEnd'] },
    ],
  });

  const canvas = document.querySelector('canvas');
  canvas.addEventListener('mousemove', (e) => {
    // console.log('e: ', e);
    if (e.buttons === 1) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgb(255,255,255)';
      ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
    }
  });

  window.clear = () => {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgb(0,0,0)';
    ctx.fillRect(0, 0, 300, 300);
  };

  clear();

  window.predict = () => {
    const input = tf.tidy(() => {
      return tf.image
        .resizeBilinear(tf.browser.fromPixels(canvas), [28, 28], true)
        .slice([0, 0, 0], [28, 28, 1])
        .toFloat()
        .div(255)
        .reshape([1, 28, 28, 1]);
    });
    const pred = model.predict(input).argMax(1);
    console.log(`预测结果`, pred.dataSync()[0]);
  };
})();
