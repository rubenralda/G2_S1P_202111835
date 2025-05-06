const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_CLASSES = 10;

async function loadImagesFromDir(dirPath, label) {
  const images = [];
  const files = fs.readdirSync(dirPath);

  for (let file of files) {
    const imageBuffer = fs.readFileSync(path.join(dirPath, file));
    const imageTensor = tf.node.decodeImage(imageBuffer, 1)
      .resizeNearestNeighbor([IMAGE_WIDTH, IMAGE_HEIGHT])
      .toFloat()
      .div(255.0)
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
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
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
  
  console.log(xs, ys)
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
