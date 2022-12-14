import * as tf from '@tensorflow/tfjs';

// const t0 = tf.tensor(1);
// const t1 = tf.tensor([1, 2, 3]);
// console.log('t1: ', t1);
// t1.print();

// const t2 = tf.tensor([[1, 2], [2], [3], [4]]);
// console.log('t2: ', t2);

const input = [1, 2, 3, 4];
const w = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
  [4, 5, 6, 7],
];
const output = [0, 0, 0, 0];
//循环权重
// for (let i = 0; i < w.length; i++) {
//   for (let j = 0; j < input.length; j++) {
//     output[i] += input[j] * w[i][j];
//   }
// }
// console.log('output: ', output);

tf.tensor(w).dot(tf.tensor(input)).print();
