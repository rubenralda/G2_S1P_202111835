const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;

async function preprocessImage(imagePath) {
  const img = await loadImage(imagePath);
  const canvas = createCanvas(IMAGE_WIDTH, IMAGE_HEIGHT);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
  const imageData = ctx.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

  return tf.browser.fromPixels(imageData, 1)
    .toFloat()
    .div(255)
    .expandDims(0);
}

async function predict(imagePath) {
  const model = await tf.loadLayersModel('file://./model/model.json');
  const imageTensor = await preprocessImage(imagePath);

  const prediction = model.predict(imageTensor);
  const result = prediction.argMax(-1).dataSync()[0];

  console.log(`El número predicho es: ${result}`);
}

predict('./some-test-image.png');