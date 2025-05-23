const main = require('./prediccionNumerica/train.js');
const predict = require('./prediccionNumerica/predict.js');
const express = require('express');
const morgan = require('morgan')
const app = express();
const multer = require('multer')
const fs = require('fs');
const cors = require('cors');

app.set('port', process.env.PORT || 3000)

app.use(cors({
  origin: 'http://localhost:9000',
  methods: ['GET', 'POST'],
  credentials: true
}));
app.use(express.json());
app.use(morgan('dev'))
app.use(express.urlencoded({extended: false}))

app.get('/train/prediccionNumerico', async (req, res) => {
    await main()
    return res.json({
        'status': true
    })
})

const storage = multer.diskStorage({
  destination: './uploads',
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});
const upload = multer({ storage });

app.post('/prediccionNumerico', upload.single('image'), async (req, res) => {
    try {
      const result = await predict(req.file.path);
      console.log(req.file.path)
      fs.unlinkSync(req.file.path); // Borra imagen temporal
  
      res.json(result);
    } catch (error) {
      console.error('Error en /prediccionNumerico:', error);
      res.status(500).json({ error: 'Error al procesar la imagen.' });
    }
  });

app.listen(app.get('port'), () => {
    console.log(`Server Listening on PORT ${app.get('port')}`);
});