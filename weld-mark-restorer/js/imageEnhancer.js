/**
 * 이미지 명료화 (브라우저 Canvas 기반, 외부 의존성 없음)
 *
 * 1) 녹(주황/갈색) 억제: 파란 채널 가중 그레이스케일 — 흰 분필/마커는 B값이 높고 녹은 낮다
 * 2) 자동 레벨(2%~98% 백분위 대비 확장)
 * 3) 언샤프 마스크로 흐릿한 획 선명화
 *
 * 나중에 서버측 전처리(예: OpenCV, 초해상도 모델)로 바꿀 경우 enhanceImage만 교체.
 */

const MAX_SIDE = 1200;

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

export async function enhanceImage(dataUrl) {
  const img = await loadImage(dataUrl);
  const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  const d = imageData.data;
  const n = w * h;

  // 1) 녹 억제 그레이스케일
  const gray = new Float32Array(n);
  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) {
    const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2];
    const v = 0.15 * r + 0.3 * g + 0.55 * b;
    gray[i] = v;
    hist[Math.min(255, v | 0)]++;
  }

  // 2) 자동 레벨
  const lo = percentile(hist, n, 0.02);
  const hi = percentile(hist, n, 0.98);
  const range = Math.max(1, hi - lo);
  for (let i = 0; i < n; i++) {
    gray[i] = clamp(((gray[i] - lo) / range) * 255);
  }

  // 3) 언샤프 마스크 (3x3 박스 블러 기준)
  const amount = 1.2;
  const out = new Uint8ClampedArray(d.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, cnt = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          sum += gray[yy * w + xx];
          cnt++;
        }
      }
      const i = y * w + x;
      const v = clamp(gray[i] + amount * (gray[i] - sum / cnt));
      out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = v;
      out[i * 4 + 3] = 255;
    }
  }

  ctx.putImageData(new ImageData(out, w, h), 0, 0);
  return canvas.toDataURL('image/jpeg', 0.9);
}

function percentile(hist, total, p) {
  const target = total * p;
  let acc = 0;
  for (let i = 0; i < 256; i++) {
    acc += hist[i];
    if (acc >= target) return i;
  }
  return 255;
}

function clamp(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}
