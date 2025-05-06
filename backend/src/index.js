const main = require('./prediccionNumerica/train.js');
const predict = require('./prediccionNumerica/predict.js');
const express = require('express');
const morgan = require('morgan')
const app = express();

app.set('port', process.env.PORT || 3000)

app.use(express.json());
app.use(morgan('dev'))
app.use(express.urlencoded({extended: false}))

app.get('/train/prediccionNumerico', async (req, res) => {
    await main()
    return res.json({
        'status': true
    })
})

// Configurar almacenamiento
const storage = multer.diskStorage({
  destination: './uploads',
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});
const upload = multer({ storage });

app.post('/prediccionNumerico', upload.single('image'), async (req, res) => {
    try {
      const result = await predict(req.file.path);
      fs.unlinkSync(req.file.path); // Borra imagen temporal
  
      res.json(result);
    } catch (error) {
      console.error('Error en /predict-number:', error);
      res.status(500).json({ error: 'Error al procesar la imagen.' });
    }
  });

app.listen(app.get('port'), () => {
    console.log(`Server Listening on PORT ${app.get('port')}`);
});