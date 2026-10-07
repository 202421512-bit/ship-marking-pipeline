/**
 * 데모용 샘플 이미지 생성: 녹/오염이 있는 철판 위에 흐릿한 분필 용접 표기.
 * 숫자 부분은 일부러 지워져 "3"인지 "8"인지 애매하게 그린다.
 */

export function createSampleImage() {
  const W = 900, H = 600;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d');
  const rand = mulberry32(42);

  // 철판 바탕
  const g = ctx.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, '#6d7277');
  g.addColorStop(0.5, '#5a5f63');
  g.addColorStop(1, '#4b4f52');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  // 금속 결 (가는 가로 스크래치)
  for (let i = 0; i < 500; i++) {
    ctx.strokeStyle = `rgba(${rand() > 0.5 ? '255,255,255' : '0,0,0'},${0.03 + rand() * 0.05})`;
    ctx.lineWidth = 1;
    const y = rand() * H, x = rand() * W;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 40 + rand() * 160, y + (rand() - 0.5) * 3);
    ctx.stroke();
  }

  // 녹 얼룩
  for (let i = 0; i < 45; i++) {
    const x = rand() * W, y = rand() * H, r = 20 + rand() * 90;
    const rg = ctx.createRadialGradient(x, y, 0, x, y, r);
    const a = 0.25 + rand() * 0.45;
    rg.addColorStop(0, `rgba(${150 + rand() * 40},${60 + rand() * 30},20,${a})`);
    rg.addColorStop(1, 'rgba(120,60,20,0)');
    ctx.fillStyle = rg;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // 분필 획
  const chalk = (pts, width = 9) => {
    for (let pass = 0; pass < 3; pass++) {
      ctx.strokeStyle = `rgba(235,235,225,${0.18 + rand() * 0.12})`;
      ctx.lineWidth = width * (0.7 + rand() * 0.5);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      pts.forEach(([x, y], i) => {
        const jx = x + (rand() - 0.5) * 4, jy = y + (rand() - 0.5) * 4;
        i ? ctx.lineTo(jx, jy) : ctx.moveTo(jx, jy);
      });
      ctx.stroke();
    }
  };

  // 기준선 + 화살표
  chalk([[300, 260], [760, 255]]);
  chalk([[300, 260], [170, 420]]);
  chalk([[170, 420], [178, 380]]);
  chalk([[170, 420], [205, 405]]);
  // 필릿 삼각형 (화살표측 = 기준선 아래)
  chalk([[400, 260], [400, 335], [470, 258]]);
  // 길이 100
  chalk([[520, 290], [520, 340]], 7);
  chalk([[545, 290], [545, 340], [575, 340], [575, 290], [545, 290]], 7);
  chalk([[595, 290], [595, 340], [625, 340], [625, 290], [595, 290]], 7);
  // W
  chalk([[330, 150], [345, 220], [362, 175], [379, 220], [394, 150]], 10);
  // 숫자: 3/8 애매 (오른쪽 반원 두 개 + 왼쪽은 희미)
  chalk([[420, 158], [445, 150], [458, 168], [440, 185]], 10);
  chalk([[440, 185], [460, 200], [450, 220], [420, 215]], 10);
  ctx.globalAlpha = 0.35;
  chalk([[420, 160], [418, 178], [438, 186]], 7);
  chalk([[438, 186], [418, 198], [420, 215]], 7);
  ctx.globalAlpha = 1;

  // 오염 / 지워짐 (분필 위에 덮인 얼룩)
  for (let i = 0; i < 18; i++) {
    const x = 300 + rand() * 450, y = 140 + rand() * 230, r = 10 + rand() * 40;
    const rg = ctx.createRadialGradient(x, y, 0, x, y, r);
    rg.addColorStop(0, `rgba(${90 + rand() * 60},${55 + rand() * 20},30,${0.35 + rand() * 0.3})`);
    rg.addColorStop(1, 'rgba(90,60,30,0)');
    ctx.fillStyle = rg;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // 노이즈
  const id = ctx.getImageData(0, 0, W, H);
  for (let i = 0; i < id.data.length; i += 4) {
    const nz = (rand() - 0.5) * 22;
    id.data[i] += nz;
    id.data[i + 1] += nz;
    id.data[i + 2] += nz;
  }
  ctx.putImageData(id, 0, 0);

  // 흐림
  const blurred = document.createElement('canvas');
  blurred.width = W;
  blurred.height = H;
  const bctx = blurred.getContext('2d');
  bctx.filter = 'blur(1.6px)';
  bctx.drawImage(c, 0, 0);
  return blurred.toDataURL('image/jpeg', 0.88);
}

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
