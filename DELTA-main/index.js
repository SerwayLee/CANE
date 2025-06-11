const express = require('express');
const bodyParser = require('body-parser');
const fs = require("fs");

const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(express.static('public'));
app.set("view engine", "ejs");

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/ab123.html');
    return res.render('ab123', {
    })
});



app.post('/receive', (req, res) => {
  const { sentence } = req.body;
  if (!sentence) return res.status(400).json({ status:'error', message:'No sentence provided' });
  console.log('[RECEIVE]', sentence);
  queue.push(sentence);
  return res.json({ status:'ok' });
});

app.get('/request', (req, res) => {
  const messages = queue.slice();
  queue = [];
  console.log('[REQUEST] sending', messages.length, 'items');
  return res.json({ sentences: messages });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
app.listen(port, () => {
    console.log(`서버가 http://localhost:${port} 에서 실행 중입니다.`);
});
