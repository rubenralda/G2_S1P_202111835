const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;

async function preprocessImage(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
    return tf.node.decodeImage(imageBuffer, 1) // escala de grises
      .resizeNearestNeighbor([size, size])
      .toFloat()
      .div(255.0)
      .expandDims(); // [1, size, size, 1]
}

async function predict(imagePath) {
  const model = await tf.loadLayersModel('file://./model/model.json');
  const imageTensor = await preprocessImage(imagePath);

  const prediction = model.predict(imageTensor);
  const probs = prediction.dataSync();
  const result = prediction.argMax(-1).dataSync()[0];

  console.log(`El número predicho es: ${result}`);

  return {probs, result}
}

module.exports = predict
//predict('./some-test-image.png');