// 의존성 없는 정적 파일 서버: `node dev-server.js` 또는 `npm start`
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PORT = Number(process.env.PORT) || 5173;
const ROOT = __dirname;
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.json': 'application/json',
};

http
  .createServer((req, res) => {
    const urlPath = decodeURIComponent(req.url.split('?')[0]);
    const file = path.normalize(path.join(ROOT, urlPath === '/' ? 'index.html' : urlPath));
    if (!file.startsWith(ROOT)) {
      res.writeHead(403).end();
      return;
    }
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404).end('Not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
      res.end(data);
    });
  })
  .listen(PORT, '0.0.0.0', () => {
    console.log(`용접 표기 복원 도우미: http://localhost:${PORT}`);
    for (const ifs of Object.values(os.networkInterfaces())) {
      for (const i of ifs || []) {
        if (i.family === 'IPv4' && !i.internal) console.log(`  휴대폰(같은 Wi-Fi): http://${i.address}:${PORT}`);
      }
    }
  });
