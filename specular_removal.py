"""
단일 영상 반사광(정반사) 분리 - Tan & Ikeuchi, "Separating Reflection Components of Textured
Surfaces Using a Single Image", IEEE TPAMI 27(2), 2005 의 수식 모델을 벡터화해 구현.

반사 모델 (이색성 반사, dichromatic):  I(x) = m_d(x)·Λ(x) + m_s(x)·Γ
  Λ: 난반사(diffuse) 색도, Γ: 광원(=정반사) 색도, m_d / m_s: 각 성분의 크기

처리 단계
  1. 조명 정규화 (Eq.10)        I' = I / (3Γ)              -> 정반사 성분이 순백색이 됨
  2. 최대 색도 (Eq.13)          σ' = max(I') / (ΣI' + ε)
  3. Specular-Free 영상 (Eq.22, 24)
                                m_d^sf = max(I')(3σ' - 1) / (σ'(3Λ_new - 1) + ε),   Λ_new = 0.5
                                I_sf  = I' - (ΣI' - m_d^sf) / 3
  4. 하이라이트 검출 (로그 미분)  Δ = |d log ΣI' - d log ΣI_sf|  > threshold
     국소 감소 (Eq.18, 29~30)   이웃 순수 난반사 픽셀의 σ' 를 Λ 로 써서
                                m_d' = max(I')(3σ' - 1) / (σ'(3Λ - 1)),  m_s' = ΣI' - m_d'
                                I_diff = I' - m_s'/3,   I_spec = I' - I_diff
     색 불연속 (Eq.31)          Δr > thR or Δg > thG 인 경계는 넘어가지 않음

논문과 다르게 구현한 부분 (속도/안정성)
  - 논문은 픽셀 쌍을 순차로 비교하며 ε 를 줄여 가는 반복 루프다. 여기서는 "이웃 순수 난반사 σ'" 를
    색 불연속 경계에서 막히는 3x3 팽창(cv2.dilate)을 반복해 한꺼번에 전파한다 (지오데식 팽창).
  - 하이라이트 판정: 로그 미분 Δ 는 하이라이트의 가장자리에서만 크다 (내부가 평평하면 양쪽 미분이
    모두 0). 그래서 Δ > threshold 에 더해, 전파된 이웃 난반사 σ' 보다 자기 σ' 가 뚜렷이 낮은
    픽셀(= 흰빛이 섞인 픽셀)도 하이라이트로 본다.
  - 색 불연속(Eq.31)은 입력 대신 SF 영상의 색도로 판단한다. 입력 색도는 하이라이트 경계에서도
    크게 바뀌어 실제 색 경계와 구분이 안 되지만, SF 영상은 정반사가 빠져 색상 경계만 남는다.

한계 (중요)
  - 무채색 표면(회색 철판, 흰 분필, 검은 잉크)은 난반사 색도가 광원 색도와 같아 이 모델로는
    정반사를 구분할 수 없다 (σ' ≈ 1/3). 이런 픽셀은 손대지 않고 그대로 둔다.
    유색 도료(노랑, 빨강, 파랑)와 프라이머(적갈색) 위의 반사광에 효과가 있다.
  - 센서 포화(255) 픽셀은 색 정보가 없으므로 주변 난반사 결과로 인페인팅한다 (복원이 아니라 추정).
"""
import cv2
import numpy as np

EPS = 1e-6


class SpecularSeparator:
    def __init__(self, target_chroma=0.5, log_diff_thresh=0.03,
                 illum_chroma=(1 / 3, 1 / 3, 1 / 3),
                 chroma_disc_thresh=(0.06, 0.06),
                 achromatic_eps=0.03, chroma_margin=(0.005, 0.02),
                 propagation_iters=30, core_iters=6, saturation_level=250, dark_level=0.06,
                 bright_level=0.6, inpaint_radius=5, analysis_scale=0.5, inpaint_scale=0.25):
        """
        target_chroma       Λ_new: SF 영상을 만들 때 쓰는 임의의 기준 최대 색도 (1/3 < Λ_new ≤ 1)
        log_diff_thresh     로그 미분 차이 Δ 의 하이라이트 판정 임계값
        illum_chroma        광원 색도 Γ (r, g, b). 기본 백색광
        chroma_disc_thresh  색 불연속 임계값 (thR, thG)
        achromatic_eps      σ' ≤ 1/3 + 이 값이면 무채색으로 보고 분리하지 않음 (약간 푸르스름한
                            회색 강판 σ'≈0.35 도 무채색으로 들어가게 0.03)
        chroma_margin       (시작, 끝): 이웃 난반사보다 σ' 가 '시작' 만큼 낮으면 보정을 시작해
                            '끝' 에서 100% 보정 (경계에 고리 모양 계단이 생기지 않게 부드럽게)
        propagation_iters   이웃 난반사 σ' 를 전파하는 3x3 팽창 횟수 (= 최대 전파 거리 px)
        core_iters          밝은 무채색(하이라이트 중심)으로 추가 전파하는 횟수. 중심은 받기만 하고
                            더 퍼뜨리지 않아, 회색 바탕의 하이라이트로 도료 색이 번지지 않음
        saturation_level    어느 채널이든 이 값 이상이면 센서 포화로 봄 (0~255)
        dark_level          ΣI'/3 가 이보다 어두우면 (0~1) 색도가 불안정하므로 손대지 않음
        inpaint_radius      포화 영역 인페인팅 반경 (인페인팅 해상도 기준 px)
        analysis_scale      하이라이트 판정(로그 미분, 색 경계, σ' 전파)을 이 배율로 줄여서 계산.
                            최종 보정(Eq.18)과 SF 영상은 원본 해상도에서 픽셀 단위로 계산한다.
        inpaint_scale       포화 영역 인페인팅 해상도 배율 (하이라이트 중심은 매끈해서 줄여도 됨)
        """
        if not 1 / 3 < target_chroma <= 1:
            raise ValueError("target_chroma 는 (1/3, 1] 범위여야 합니다")
        gamma = np.asarray(illum_chroma, np.float32)
        if gamma.shape != (3,) or np.any(gamma <= 0):
            raise ValueError("illum_chroma 는 양수 3개 (r, g, b) 여야 합니다")
        gamma = gamma / gamma.sum()
        self.illum_scale_bgr = 3 * gamma[::-1]   # Eq.10: I' = I / (3Γ), 채널 순서 BGR
        self.target_chroma = float(target_chroma)
        self.log_diff_thresh = float(log_diff_thresh)
        self.th_r, self.th_g = chroma_disc_thresh
        self.achromatic_eps = float(achromatic_eps)
        self.margin_lo, self.margin_hi = map(float, chroma_margin)
        self.propagation_iters = int(propagation_iters)
        self.core_iters = int(core_iters)
        self.bright_sum = 3 * float(bright_level)
        self.saturation_level = saturation_level
        self.dark_sum = 3 * float(dark_level)
        self.inpaint_radius = inpaint_radius
        self.analysis_scale = float(analysis_scale)
        self.inpaint_scale = float(inpaint_scale)
        self.masks = {}  # 마지막 호출의 중간 마스크 (디버깅/시각화용)

    # ------------------------------------------------------------------
    # 수식 (Eq.13, 22/24, 18)
    # ------------------------------------------------------------------
    @staticmethod
    def max_chromaticity(img):
        """Eq.13: σ' = max(I') / (ΣI' + ε). (σ', ΣI', max I') 반환."""
        b, g, r = cv2.split(img)   # 채널 분리 후 더하는 편이 img.sum(axis=2) 보다 훨씬 빠름
        total = b + g + r
        imax = np.maximum(np.maximum(b, g), r)
        return imax / (total + EPS), total, imax

    @staticmethod
    def diffuse_scale(imax, sigma, lam):
        """Eq.22: m_d = max(I')(3σ' - 1) / (σ'(3Λ - 1) + ε). 무채색(σ'→1/3)이면 0 으로 수렴."""
        num = sigma * 3 - 1
        num *= imax
        den = np.asarray(lam, np.float32) * 3 - 1
        den *= sigma
        den += EPS
        return np.divide(num, den, out=num)

    @staticmethod
    def specular_amount(total, md, weight=None):
        """Eq.18: m_s = ΣI' - m_d (≥ 0). weight 가 있으면 보정 강도를 곱한다."""
        ms = np.subtract(total, md)
        np.maximum(ms, 0, out=ms)
        if weight is not None:
            ms *= weight
        return ms

    @staticmethod
    def remove_specular(img, ms, clip=True):
        """I_diff = I' - m_s/3: 백색 정반사를 채널마다 똑같이 뺀다 (OpenCV 한 번에 3채널)."""
        third = ms * (1 / 3)
        out = cv2.subtract(img, cv2.merge((third, third, third)))
        if clip:
            np.maximum(out, 0, out=out)
        return out

    @staticmethod
    def _diff_xy(a):
        """전방 차분 (x, y). 마지막 열/행은 0."""
        dx = np.zeros_like(a)
        dy = np.zeros_like(a)
        dx[:, :-1] = a[:, 1:] - a[:, :-1]
        dy[:-1, :] = a[1:, :] - a[:-1, :]
        return dx, dy

    @staticmethod
    def _geodesic_max(ref, allowed, iters):
        """allowed 영역 안에서만 3x3 최대값 팽창을 iters 번. 막힌 픽셀은 값을 받지도 넘기지도 않음."""
        gate = allowed.astype(np.float32)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(iters):
            grown = cv2.dilate(ref, kernel)
            np.multiply(grown, gate, out=grown)   # 불리언 인덱싱보다 빠름
            np.maximum(ref, grown, out=ref)
        return ref

    def _pair_mask(self, dx_hit, dy_hit):
        """차분이 걸린 픽셀 쌍의 양쪽 픽셀을 모두 표시."""
        m = dx_hit | dy_hit
        m[:, 1:] |= dx_hit[:, :-1]
        m[1:, :] |= dy_hit[:-1, :]
        return m

    # ------------------------------------------------------------------
    def _analyze(self, img, saturated):
        """하이라이트 판정 (축소 해상도). 이웃 난반사 σ' 기준값 ref, 로그 미분 하이라이트 구역 등 반환."""
        sigma, total, imax = self.max_chromaticity(img)
        dark = total < self.dark_sum
        chromatic = sigma > 1 / 3 + self.achromatic_eps

        # 3) SF 영상 (Eq.22, 24)
        md_sf = self.diffuse_scale(imax, sigma, self.target_chroma)
        sf = self.remove_specular(img, self.specular_amount(total, md_sf))
        _, sf_total, _ = self.max_chromaticity(sf)

        # 색 불연속 (Eq.31) - SF 영상의 r, g 색도로 판단 (모듈 docstring 참고)
        r = sf[..., 2] / (sf_total + EPS)
        g = sf[..., 1] / (sf_total + EPS)
        rdx, rdy = self._diff_xy(r)
        gdx, gdy = self._diff_xy(g)
        edge = self._pair_mask((np.abs(rdx) > self.th_r) | (np.abs(gdx) > self.th_g),
                               (np.abs(rdy) > self.th_r) | (np.abs(gdy) > self.th_g))
        edge &= chromatic & ~dark  # 무채색/암부의 색도 잡음은 경계로 치지 않음

        # 4-a) 로그 미분 (Eq.29~30): 순수 난반사면 d log I' = d log I_sf
        ldx, ldy = self._diff_xy(np.log(total + EPS))
        sdx, sdy = self._diff_xy(np.log(sf_total + EPS))
        delta = np.maximum(np.abs(ldx - sdx), np.abs(ldy - sdy))
        log_hit = (delta > self.log_diff_thresh) & ~dark & ~edge

        # 4-b) 이웃 순수 난반사 σ' 전파 (3x3 지오데식 팽창)
        #   유채색 픽셀 사이로만 퍼지고 색 경계·무채색(회색 강판)·암부에서 막힌다.
        #   밝은 무채색 = 흰빛이 대부분인 하이라이트 중심은 마지막에 core_iters 만큼 받기만 한다.
        scale = img.shape[1] / self._full_width
        iters = lambda n: max(1, int(np.ceil(n * scale)))
        passable = chromatic & ~dark & ~edge
        source = passable & ~saturated & ~log_hit
        smooth = cv2.GaussianBlur(sigma, (5, 5), 0)  # 잡음 최대값이 퍼지지 않게
        ref = np.where(source, smooth, 0).astype(np.float32)
        ref = self._geodesic_max(ref, passable, iters(self.propagation_iters))
        core = ~chromatic & (total > self.bright_sum)
        ref = self._geodesic_max(ref, passable | core, iters(self.core_iters))
        log_zone = cv2.dilate(log_hit.astype(np.uint8), np.ones((3, 3), np.uint8))
        return ref, log_zone, edge, delta

    def separate(self, image_bgr):
        """BGR uint8 -> (diffuse_bgr, specular_bgr, specular_free_bgr), 모두 uint8."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("3채널 BGR 이미지가 필요합니다")
        h, w = image_bgr.shape[:2]
        self._full_width = w

        # 1) 조명 정규화 (Eq.10)
        norm = tuple(float(v) for v in 1 / (255 * self.illum_scale_bgr)) + (0.0,)
        img = cv2.multiply(image_bgr, norm, dtype=cv2.CV_32F)

        # 2) 최대 색도 (Eq.13)
        sigma, total, imax = self.max_chromaticity(img)
        cb, cg, cr = cv2.split(image_bgr)
        saturated = cv2.max(cv2.max(cb, cg), cr) >= self.saturation_level
        dark = total < self.dark_sum
        chromatic = sigma > 1 / 3 + self.achromatic_eps

        # 3) Specular-Free 영상 (Eq.22, 24) - 출력용, 원본 해상도
        md_sf = self.diffuse_scale(imax, sigma, self.target_chroma)
        sf = self.remove_specular(img, self.specular_amount(total, md_sf), clip=False)

        # 4-a, b) 하이라이트 판정은 축소 해상도에서 (판정 값들은 매끈해서 줄여도 됨)
        f = self.analysis_scale if min(h, w) * self.analysis_scale >= 64 else 1.0
        if f < 1:
            small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            sat_small = cv2.resize(saturated.astype(np.uint8), (small.shape[1], small.shape[0]),
                                   interpolation=cv2.INTER_AREA) > 0
            ref, log_zone, edge, delta = self._analyze(small, sat_small)
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
            log_zone = cv2.resize(log_zone, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            ref, log_zone, edge, delta = self._analyze(img, saturated)

        # 하이라이트: Δ 가 큰 곳 + 이웃 난반사보다 σ' 가 낮은 곳 (흰빛이 섞임). 원본 해상도.
        deficit = ref - sigma
        weight = np.clip((deficit - self.margin_lo) * (1 / (self.margin_hi - self.margin_lo)), 0, 1)
        weight[(log_zone > 0) & (deficit > 0)] = 1
        core = ~chromatic & (total > self.bright_sum)
        weight[(ref <= 1 / 3 + self.achromatic_eps) | dark | ~(chromatic | core)] = 0

        # 4-c) 국소 감소 (Eq.18): 이웃 난반사 σ' 를 Λ 로 써서 m_d, m_s 재계산
        corrected = weight > 0
        lam = np.where(corrected, ref, sigma)
        md = self.diffuse_scale(imax, sigma, lam)
        diffuse = self.remove_specular(img, self.specular_amount(total, md, weight), clip=False)

        # 되돌리기: 정규화 해제 -> 원래 광원 아래의 색. CV_8U 출력은 반올림 + 0~255 포화 처리
        back = tuple(float(v) for v in 255 * self.illum_scale_bgr) + (0.0,)
        diffuse_u8 = cv2.multiply(diffuse, back, dtype=cv2.CV_8U)
        sf_u8 = cv2.multiply(sf, back, dtype=cv2.CV_8U)

        # 포화 픽셀: 색 정보가 없어 수식으로 못 구하므로 주변 난반사로 인페인팅 (축소 해상도)
        if saturated.any():
            kernel = np.ones((3, 3), np.uint8)
            hole = cv2.dilate(saturated.astype(np.uint8), kernel)
            fi = self.inpaint_scale if min(h, w) * self.inpaint_scale >= 32 else 1.0
            small = cv2.resize(diffuse_u8, None, fx=fi, fy=fi, interpolation=cv2.INTER_AREA)
            hole_small = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
            hole_small = cv2.dilate((hole_small > 0).astype(np.uint8), kernel)
            filled = cv2.inpaint(small, hole_small, self.inpaint_radius, cv2.INPAINT_TELEA)
            filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
            hole_b = hole.astype(bool)
            diffuse_u8[hole_b] = filled[hole_b]
            # 정반사는 빛을 더하기만 하므로 난반사는 관측값을 넘을 수 없다 (인페인팅 추정이 넘치는 채널 제한)
            diffuse_u8 = cv2.min(diffuse_u8, image_bgr)

        specular_u8 = cv2.subtract(image_bgr, diffuse_u8)

        self.masks = {"highlight": corrected | saturated, "corrected": corrected, "saturated": saturated,
                      "achromatic": ~chromatic,
                      "edge": edge, "log_diff": delta}  # edge, log_diff 는 analysis_scale 해상도
        return diffuse_u8, specular_u8, sf_u8


# ----------------------------------------------------------------------
# 합성 검증 영상: 정답 난반사를 아는 상태에서 백색 정반사를 얹는다
# ----------------------------------------------------------------------
def make_synthetic_scene(width=1024, height=768, seed=0):
    """유색 도료 패치 4개 + 회색 강판 배경에 Lambert 음영과 Phong 하이라이트를 얹은 장면.

    (입력 BGR, 정답 난반사 BGR, 정반사 강도 맵 0~1) 반환.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    albedo = np.full((height, width, 3), (0.42, 0.43, 0.45), np.float32)      # 회색 강판 (BGR)
    paints = [((0.10, 0.75, 0.90), (0.05, 0.05, 0.45, 0.45)),                  # 노랑 도료
              ((0.15, 0.20, 0.80), (0.55, 0.05, 0.95, 0.45)),                  # 빨강 도료
              ((0.75, 0.45, 0.10), (0.05, 0.55, 0.45, 0.95)),                  # 파랑 도료
              ((0.25, 0.35, 0.55), (0.55, 0.55, 0.95, 0.95))]                  # 적갈색 프라이머
    for color, (x0, y0, x1, y1) in paints:
        sl = (slice(int(y0 * height), int(y1 * height)), slice(int(x0 * width), int(x1 * width)))
        albedo[sl] = color
    albedo *= 1 + rng.normal(0, 0.02, (height, width, 1)).astype(np.float32)   # 도막 질감

    # 완만한 곡면 음영 + 하이라이트 (광원 몇 개)
    shade = 0.75 + 0.25 * np.cos((xx / width - 0.5) * 2.5) * np.cos((yy / height - 0.5) * 2.0)
    spec = np.zeros((height, width), np.float32)
    for cx, cy, r, k in [(0.25, 0.25, 0.06, 1.0), (0.75, 0.25, 0.05, 0.8),
                         (0.25, 0.75, 0.07, 0.9), (0.75, 0.75, 0.05, 0.7), (0.5, 0.5, 0.06, 0.9)]:
        d2 = ((xx - cx * width) ** 2 + (yy - cy * height) ** 2) / (r * width) ** 2
        spec += k * np.exp(-d2)
    diffuse = np.clip(albedo * shade[..., None], 0, 1)
    observed = np.clip(diffuse + spec[..., None] * 0.85, 0, 1)                  # 백색 정반사
    to_u8 = lambda a: (a * 255 + 0.5).astype(np.uint8)
    return to_u8(observed), to_u8(diffuse), spec


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (8 + 11 * len(text), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Tan & Ikeuchi (2005) 반사광 분리")
    parser.add_argument("image", nargs="?", help="입력 이미지 (없으면 합성 장면)")
    parser.add_argument("-o", "--output", default="specular_result.jpg")
    parser.add_argument("--thresh", type=float, default=0.03, help="로그 미분 임계값")
    args = parser.parse_args()

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {args.image}")
        truth = None
    else:
        bgr, truth, _ = make_synthetic_scene()

    sep = SpecularSeparator(log_diff_thresh=args.thresh)
    sep.separate(bgr)  # 첫 호출 워밍업
    start = time.perf_counter()
    diffuse, specular, sf = sep.separate(bgr)
    ms = (time.perf_counter() - start) * 1000
    print(f"{bgr.shape[1]}x{bgr.shape[0]}: {ms:.1f} ms, 보정 픽셀 {sep.masks['corrected'].mean() * 100:.1f}%, "
          f"포화 {sep.masks['saturated'].mean() * 100:.1f}%, 무채색 {sep.masks['achromatic'].mean() * 100:.1f}%")
    if truth is not None:
        hl = sep.masks["highlight"]
        err = lambda a: np.abs(a.astype(np.int16) - truth).mean(axis=2)[hl].mean()
        print(f"하이라이트 영역 정답 대비 평균 오차: 입력 {err(bgr):.1f} -> 난반사 {err(diffuse):.1f} (0~255)")

    top = np.hstack([_label(bgr, "input"), _label(diffuse, "diffuse")])
    bottom = np.hstack([_label(specular, "specular"), _label(sf, "specular-free (SF)")])
    cv2.imwrite(args.output, np.vstack([top, bottom]))
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
