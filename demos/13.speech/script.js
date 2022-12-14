import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';
const MODEL_PATH = 'http://127.0.0.1:8080/speech';

(async () => {
  //BROWSER_FFT
  //傅里叶变换 任何周期函数都都可以看作是不同振幅 不同相位正弦波的叠加
  //傅里叶变换是一种预处理方式，将一个频谱分解为多个频谱，这些频谱可以被用来表示一个信号的频率特征
  //频谱是一个二维的矩阵，每一行代表一个频率，每一列代表一个时间点
  //傅立叶变换，表示能将满足一定条件的某个函数表示成三角函数（正弦和/或余弦函数）或者它们的积分的线性组合。
  const recognizer = speechCommands.create(
    'BROWSER_FFT',
    null,
    MODEL_PATH + '/model.json',
    MODEL_PATH + '/metadata.json'
  );

  await recognizer.ensureModelLoaded();
  const labels = recognizer.wordLabels().splice(2);
  console.log(labels);
  const resultEl = document.querySelector('#result');
  resultEl.innerHTML = labels
    .map(
      (l) => `
  <div>${l}</div>
`
    )
    .join('');
  recognizer.listen(
    (result) => {
      const { scores } = result;
      //   console.log('scores: ', scores);
      const maxValue = Math.max(...scores);
      const index = scores.indexOf(maxValue) - 2;
      resultEl.innerHTML = labels
        .map(
          (l, i) => `
    <div style="background: ${i === index && 'green'}">${l}</div>
    `
        )
        .join('');
    },
    {
      overlapFactor: 0.3,
      probabilityThreshold: 0.9,
    }
  );
})();
