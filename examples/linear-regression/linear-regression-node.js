const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

tf.setBackend('tensorflow');

async function myFirstTfjs() {
	// Create a simple model.
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 1, inputShape: [1]}));

	// Prepare the model for training: Specify the loss and the optimizer.
	model.compile({
		loss: 'meanSquaredError',
		optimizer: 'sgd'
	});

	// Generate some synthetic data for training. (y = 2x - 1)
	const xs = tf.tensor1d([1, 2, 3, 4, 5]);
	const ys = tf.tensor1d([2, 4, 6, 8, 10]);

	// Train the model using the data.
	await model.fit(xs, ys, {epochs: 50});

	// Use the model to do inference on a data point the model hasn't seen.
	// Should print approximately 39.
	const output = model.predict(tf.tensor2d([20], [1, 1])).toString()
	console.log(output);
}

myFirstTfjs();