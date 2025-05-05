const express = require('express');
const morgan = require('morgan')
const app = express();

app.set('port', process.env.PORT || 3000)

app.use(express.json());
app.use(morgan('dev'))
app.use(express.urlencoded({extended: false}))

app.listen(app.get('port'), () => {
    console.log(`Server Listening on PORT ${app.get('port')}`);
});