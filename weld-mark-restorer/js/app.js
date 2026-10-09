import { analyzeWeldMark, describeCandidate, buildWorkOrder } from './aiService.js';
import { enhanceImage } from './imageEnhancer.js';
import { createSampleImage } from './sampleImage.js';
import { renderWeldSymbolSVG } from './weldSymbol.js';

const $ = (sel) => document.querySelector(sel);

const state = {
  originalDataUrl: null,
  enhancedDataUrl: null,
  candidates: [],
  selectedId: null, // 작업자가 직접 선택하기 전까지 null — AI가 자동 확정하지 않음
  selectedAt: null,
  checkedSteps: new Set(),
};

/* ---------- 화면 전환 ---------- */
function show(id) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === id));
  document.querySelector(`#${id} .scroll`)?.scrollTo({ top: 0 });
}

/* ---------- 이미지 입력 ---------- */
function readFile(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

async function onFileChosen(e) {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file) return;
  if (!file.type.startsWith('image/')) return toast('이미지 파일을 선택하세요');
  startPipeline(await readFile(file));
}

$('#input-camera').addEventListener('change', onFileChosen);
$('#input-file').addEventListener('change', onFileChosen);
$('#btn-sample').addEventListener('click', () => startPipeline(createSampleImage()));
$('#btn-change-photo').addEventListener('click', () => show('screen-upload'));

/* ---------- 분석 파이프라인: 명료화 → AI 후보 생성 ---------- */
async function startPipeline(dataUrl) {
  state.originalDataUrl = dataUrl;
  state.selectedId = null;
  setLoading(true, 0);
  try {
    state.enhancedDataUrl = await enhanceImage(dataUrl);
    $('#img-original').src = state.originalDataUrl;
    $('#img-enhanced').src = state.enhancedDataUrl;
    await runAnalysis();
    show('screen-result');
  } catch (err) {
    console.error(err);
    setLoading(false);
    toast('이미지를 처리하지 못했습니다. 다시 시도하세요');
  }
}

async function runAnalysis() {
  state.selectedId = null;
  setLoading(true, 1);
  let step = 1;
  const timer = setInterval(() => setLoading(true, Math.min(++step, 3)), 650);
  try {
    const result = await analyzeWeldMark({
      originalDataUrl: state.originalDataUrl,
      enhancedDataUrl: state.enhancedDataUrl,
    });
    state.candidates = result.candidates;
    renderCandidates();
  } finally {
    clearInterval(timer);
    setLoading(false);
  }
}

$('#btn-reanalyze').addEventListener('click', async () => {
  try {
    await runAnalysis();
    toast('다시 분석했습니다');
  } catch (err) {
    console.error(err);
    toast('분석에 실패했습니다');
  }
});

/* ---------- 로딩 ---------- */
function setLoading(on, activeStep = 0) {
  $('#loading').hidden = !on;
  document.querySelectorAll('#loading-steps li').forEach((li) => {
    const s = Number(li.dataset.step);
    li.classList.toggle('done', s < activeStep);
    li.classList.toggle('active', s === activeStep);
  });
}

/* ---------- 전/후 비교 ---------- */
function setCompare(pct) {
  $('#compare-range').value = pct;
  $('#compare-top').style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  $('#compare-handle').style.left = `${pct}%`;
  document.querySelectorAll('.segment button').forEach((b) =>
    b.classList.toggle('on', Number(b.dataset.view) === Number(pct)),
  );
}
$('#compare-range').addEventListener('input', (e) => setCompare(Number(e.target.value)));
document.querySelectorAll('.segment button').forEach((b) =>
  b.addEventListener('click', () => setCompare(Number(b.dataset.view))),
);
setCompare(50);

/* ---------- 후보 렌더링 / 선택 ---------- */
function renderCandidates() {
  const anySelected = state.selectedId !== null;
  $('#candidates').innerHTML = state.candidates
    .map((c, i) => {
      const selected = c.id === state.selectedId;
      const emphasized = selected || (!anySelected && i === 0);
      const pct = Math.round(c.confidence * 100);
      return `
      <article class="cand ${emphasized ? 'emph' : ''} ${selected ? 'selected' : ''}" data-id="${c.id}">
        <div class="cand-head">
          <span class="cand-label">후보 ${i + 1}${selected ? '<span class="check">✓ 선택됨</span>' : ''}</span>
          <span class="pill ${i === 0 ? 'pill-strong' : 'pill-soft'}">신뢰도 ${pct}%</span>
        </div>
        <div class="symbol-box">${renderWeldSymbolSVG(c)}</div>
        <p class="cand-desc"><b>해석:</b> ${describeCandidate(c)}</p>
        <button type="button" class="btn btn-lg ${emphasized ? 'btn-primary' : 'btn-outline'}" data-select="${c.id}">
          ${selected ? '선택됨' : '이 후보 선택'}
        </button>
      </article>`;
    })
    .join('');

  $('#btn-confirm').disabled = !anySelected;
}

$('#candidates').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-select]');
  if (!btn) return;
  state.selectedId = btn.dataset.select;
  renderCandidates();
  navigator.vibrate?.(15);
});

$('#btn-confirm').addEventListener('click', () => {
  const c = state.candidates.find((x) => x.id === state.selectedId);
  if (!c) return toast('후보를 먼저 선택하세요');
  state.selectedAt = new Date();
  state.checkedSteps.clear();
  renderWorkOrder(c);
  show('screen-order');
});

/* ---------- 작업 지시 ---------- */
function renderWorkOrder(c) {
  const order = buildWorkOrder(c);
  $('#order-symbol').innerHTML = renderWeldSymbolSVG(c);
  $('#order-summary').textContent = order.summary;
  $('#order-conf').textContent = `AI 신뢰도 ${Math.round(c.confidence * 100)}%`;
  $('#order-time').textContent = state.selectedAt.toLocaleString('ko-KR', {
    month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
  $('#order-steps').innerHTML = order.steps
    .map(
      (s, i) => `
      <li>
        <button type="button" class="step" data-step="${i}" aria-pressed="false">
          <span class="num">${i + 1}</span>
          <span class="txt">${s}</span>
          <span class="tick" aria-hidden="true">✓</span>
        </button>
      </li>`,
    )
    .join('');
  updateProgress(order.steps.length);
}

function updateProgress(total) {
  const n = state.checkedSteps.size;
  $('#order-progress').textContent = `${n}/${total} 확인`;
  $('#btn-done').disabled = n < total;
}

$('#order-steps').addEventListener('click', (e) => {
  const btn = e.target.closest('.step');
  if (!btn) return;
  const i = Number(btn.dataset.step);
  state.checkedSteps.has(i) ? state.checkedSteps.delete(i) : state.checkedSteps.add(i);
  const on = state.checkedSteps.has(i);
  btn.classList.toggle('checked', on);
  btn.setAttribute('aria-pressed', String(on));
  updateProgress(document.querySelectorAll('#order-steps .step').length);
});

$('#btn-back').addEventListener('click', () => show('screen-result'));
$('#btn-home').addEventListener('click', () => {
  state.selectedId = null;
  show('screen-upload');
});
$('#btn-done').addEventListener('click', () => toast('작업 확인이 기록되었습니다'));

/* ---------- 토스트 ---------- */
let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 1800);
}
