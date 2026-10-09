/**
 * 후보 데이터 → 복원된 용접 기호 SVG (AWS/ISO 기호 단순화)
 * AI가 추정한 글자(uncertainIndex)는 강조색 + 점선 박스로 표시한다.
 */

export function renderWeldSymbolSVG(c, accent = '#007aff') {
  const ink = '#1c1c1e';
  const below = c.side !== 'other'; // 화살표측 = 기준선 아래
  const refY = 70;
  const triY = below ? refY + 32 : refY - 32;
  const lenY = below ? refY + 28 : refY - 10;

  const chars = [...c.mark];
  const widthOf = (ch) => (/[WM]/.test(ch) ? 30 : 22);
  let cursor = 110;
  const markSpans = chars
    .map((ch, i) => {
      const charW = widthOf(ch);
      const x = cursor;
      cursor += charW + 3;
      const uncertain = c.uncertainIndex?.includes(i);
      const box = uncertain
        ? `<rect x="${x - 2}" y="18" width="${charW + 2}" height="34" rx="5" fill="${accent}14" stroke="${accent}" stroke-width="1.5" stroke-dasharray="4 3"/>`
        : '';
      return `${box}<text x="${x + charW / 2 - 1}" y="45" text-anchor="middle" font-size="28" font-weight="700" fill="${uncertain ? accent : ink}">${ch}</text>`;
    })
    .join('');

  const triangle =
    c.weldType === 'fillet'
      ? `<path d="M150 ${refY} L150 ${triY} L176 ${refY} Z" fill="none" stroke="${ink}" stroke-width="3" stroke-linejoin="round"/>`
      : '';

  return `
<svg viewBox="0 0 300 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="복원된 용접 기호: ${c.mark} ${c.weldTypeLabel} ${c.sideLabel} ${c.lengthMm}mm" font-family="-apple-system, 'SF Pro Display', 'Apple SD Gothic Neo', sans-serif">
  ${markSpans}
  <line x1="92" y1="${refY}" x2="270" y2="${refY}" stroke="${ink}" stroke-width="3" stroke-linecap="round"/>
  <line x1="92" y1="${refY}" x2="38" y2="118" stroke="${ink}" stroke-width="3" stroke-linecap="round"/>
  <path d="M38 118 L42 100 L52 108 Z" fill="${ink}" stroke="${ink}" stroke-width="2" stroke-linejoin="round"/>
  ${triangle}
  <text x="190" y="${lenY}" font-size="20" font-weight="600" fill="${ink}">${c.lengthMm}</text>
</svg>`;
}
