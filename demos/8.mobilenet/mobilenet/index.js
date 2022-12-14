import * as tf from '@tensorflow/tfjs';

import { IMAGENET_CLASSES } from './imagenet_classes';

import { file2img } from './utils';

(async () => {
  const model = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  );
  window.predict = async (file) => {
    const img = await file2img(file);
    document.body.appendChild(img);
    //优化部分内存
    const pred = tf.tidy(() => {
      //将图片转换为张量
      const input = tf.browser
        .fromPixels(img)
        .toFloat()
        //归一化操作
        .sub(255 / 2)
        .div(255 / 2)
        .reshape([1, 224, 224, 3]);
      return model.predict(input);
    });
    const index = pred.argMax(1).dataSync()[0];
    console.log('index: ', index);
    setTimeout(() => {
      console.log(`预测结果`, IMAGENET_CLASSES[index]);
    }, 0);
  };
})();
