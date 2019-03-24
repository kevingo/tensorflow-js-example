const linearModel = tf.sequential();

linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));
linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([2, 4, 6, 8, 10]);

linearModel.fit(xs, ys, {epochs: 1000});

console.log('Model trained.');

linearModel.predict(tf.tensor1d([6])).print();