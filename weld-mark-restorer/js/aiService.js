/**
 * AI 분석 서비스 (현재: mock)
 *
 * 나중에 OCR / VLM / 이미지 분석 API를 연결할 때는 이 파일만 교체하면 된다.
 * UI(app.js)는 아래 함수의 입력/출력 형식에만 의존한다.
 *
 *   analyzeWeldMark(input) -> Promise<AnalysisResult>
 *
 *   input: {
 *     originalDataUrl: string,   // 업로드 원본 (data URL)
 *     enhancedDataUrl: string,   // 명료화된 이미지 (data URL)
 *   }
 *
 *   AnalysisResult: {
 *     candidates: Candidate[],   // 신뢰도 내림차순, UI는 상위 2개를 표시
 *     model: string,
 *     analyzedAt: string,        // ISO 시간
 *   }
 *
 *   Candidate: {
 *     id: string,
 *     mark: string,              // 예: "W3"
 *     weldType: "fillet" | "butt" | "plug" | ...,
 *     weldTypeLabel: string,     // 예: "필릿 용접"
 *     side: "arrow" | "other" | "both",
 *     sideLabel: string,         // 예: "화살표측"
 *     lengthMm: number,
 *     confidence: number,        // 0 ~ 1
 *     uncertainIndex: number[],  // mark 문자열 중 AI가 추정(복원)한 글자의 인덱스
 *   }
 *
 * 중요: 이 서비스는 "후보"만 제시한다. 최종 선택은 반드시 작업자가 UI에서 한다.
 */

const MOCK_CANDIDATES = [
  {
    id: 'c1',
    mark: 'W3',
    weldType: 'fillet',
    weldTypeLabel: '필릿 용접',
    side: 'arrow',
    sideLabel: '화살표측',
    lengthMm: 100,
    confidence: 0.87,
    uncertainIndex: [1],
  },
  {
    id: 'c2',
    mark: 'W8',
    weldType: 'fillet',
    weldTypeLabel: '필릿 용접',
    side: 'arrow',
    sideLabel: '화살표측',
    lengthMm: 100,
    confidence: 0.74,
    uncertainIndex: [1],
  },
];

const MOCK_LATENCY_MS = 2200;

export async function analyzeWeldMark(input) {
  // TODO: 실제 API 연결 예시
  // const res = await fetch('/api/analyze', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ image: input.enhancedDataUrl }),
  // });
  // const data = await res.json();
  // return normalize(data);

  void input;
  await new Promise((r) => setTimeout(r, MOCK_LATENCY_MS));
  return {
    candidates: MOCK_CANDIDATES
      .map((c) => ({ ...c }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 2),
    model: 'mock-v0',
    analyzedAt: new Date().toISOString(),
  };
}

/** 후보 → 사람이 읽는 해석 문장 */
export function describeCandidate(c) {
  return `${c.mark} ${c.weldTypeLabel}, ${c.sideLabel}, 길이 ${c.lengthMm} mm`;
}

/** 작업자가 선택한 후보 → 작업 지시 목록 */
export function buildWorkOrder(c) {
  const sideStep =
    c.side === 'arrow'
      ? '화살표가 가리키는 접합 위치를 확인하세요.'
      : c.side === 'other'
        ? '화살표 반대측 접합 위치를 확인하세요.'
        : '화살표측과 반대측 접합 위치를 모두 확인하세요.';
  return {
    summary: `${c.mark} / ${c.weldTypeLabel} / ${c.sideLabel} / 길이 ${c.lengthMm} mm`,
    steps: [
      sideStep,
      `해당 위치에 ${c.weldTypeLabel}을 수행하세요.`,
      `용접 길이는 ${c.lengthMm} mm입니다.`,
      '작업 전 도면 및 현장 위치와 일치하는지 확인하세요.',
    ],
  };
}
