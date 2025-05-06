const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_CLASSES = 10;

async function loadImagesFromDir(dirPath, label) {
  const files = fs.readdirSync(dirPath);
  const images = [];

  for (let file of files) {
    /* const imgPath = path.join(dirPath, file);
    const img = await loadImage(imgPath);
    const canvas = createCanvas(IMAGE_WIDTH, IMAGE_HEIGHT);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
    const imageData = ctx.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

    const pixels = tf.browser.fromPixels(imageData, 1).toFloat().div(255); */
    const imageBuffer = fs.readFileSync(path.join(classPath, file));
          const imageTensor = tf.node.decodeImage(imageBuffer, 1)
            .resizeNearestNeighbor([IMAGE_WIDTH, IMAGE_HEIGHT])
            .toFloat()
            .div(255.0)
            .expandDims();
    images.push({ tensor: imageTensor, label });
  }

  return images;
}

async function loadDataset(datasetPath) {
  const classes = fs.readdirSync(datasetPath);
  const data = [];

  for (let label of classes) {
    const labelDir = path.join(datasetPath, label);
    const images = await loadImagesFromDir(labelDir, parseInt(label));
    data.push(...images);
  }

  tf.util.shuffle(data);

  const xs = tf.stack(data.map(d => d.tensor));
  const labels = tf.tensor1d(data.map(d => d.label), 'int32');
  const ys = tf.oneHot(labels, NUM_CLASSES);

  return { xs, ys };
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, 1],
    kernelSize: 3,
    filters: 32,
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 64,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

async function main() {
  const datasetPath = path.join(__dirname, 'data');
  const { xs, ys } = await loadDataset(datasetPath);

  const model = createModel();
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 2 })
  });

  await model.save('file://./model');
  console.log('Modelo entrenado y guardado en ./model');
}

module.exports = main
