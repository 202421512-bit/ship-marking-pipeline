# Ship Marking Pipeline

조선소 철판 이미지에서 노이즈를 제거하고 정형/비정형 마킹을 필터링·분류하는 OpenCV/Python 파이프라인 데모입니다.

![CI](https://github.com/202421512-bit/ship-marking-pipeline/actions/workflows/ci.yml/badge.svg)

## 저장소 구조

```
ship-marking-pipeline/
├── .github/workflows/ci.yml   # GitHub Actions: 의존성 설치 후 pytest 실행
├── tests/test_pipeline.py     # 파이프라인 검증 테스트
├── requirements.txt           # opencv-python-headless, numpy, pytest
└── README.md
```

## 처음 세팅 (한 번만)

1. GitHub에서 저장소 초대(Collaborator)를 **Accept** 합니다.
2. Git, Python 3.11 이상, VS Code(Python 확장)를 설치합니다.
3. Git 사용자 정보를 설정합니다.
   ```bash
   git config --global user.name "이름"
   git config --global user.email "깃허브이메일@example.com"
   ```
4. 저장소를 받습니다. VS Code에서 `Ctrl+Shift+P` → **Git: Clone** 에 아래 주소를 붙여넣어도 됩니다.
   ```bash
   git clone https://github.com/202421512-bit/ship-marking-pipeline.git
   cd ship-marking-pipeline
   ```
   처음 push할 때 GitHub 로그인 창이 뜨면 브라우저로 로그인하면 됩니다.
5. 가상환경을 만들고 패키지를 설치한 뒤 테스트를 돌립니다.
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   pytest -v
   ```
   `3 passed` 가 나오면 세팅 완료입니다. VS Code 오른쪽 아래 Python 인터프리터를 `.venv` 로 선택해 두세요.

> OpenCV는 `opencv-python-headless` 만 사용합니다. `opencv-python` 을 같이 설치하면 충돌하니 섞어 쓰지 마세요.

## 작업 흐름

1. 작업 전에 최신 코드 받기
   ```bash
   git checkout main
   git pull
   ```
2. 내 브랜치 만들기 (예: `minsu/denoise`)
   ```bash
   git checkout -b 이름/작업내용
   ```
3. 수정 → 로컬에서 `pytest -v` 확인 → 커밋 → push
   ```bash
   git add .
   git commit -m "작업 내용 요약"
   git push -u origin 이름/작업내용
   ```
4. GitHub에서 **Pull Request** 를 만들고, CI가 초록불이면 merge 합니다.

`main` 에 직접 push하지 말고 항상 브랜치 → PR 로 올려 주세요.

## 마킹 분류 파이프라인

| 단계 | 내용 | 코드 |
|---|---|---|
| 1 | 정형 / 비정형 분류 | (예정) |
| 2 | 정형을 손상 / 비손상으로 구분 | (예정) |
| 3 | 비정형에 필터를 적용해 "심하게 손상된 정형"일 가능성(score 0~1) 계산 | (예정) |
| 4 | score가 높은 비정형을 정형(손상) 그룹으로 복귀 | `marking_pipeline/step4_return.py` |

4단계 기준: `score >= 0.7` 정형(손상)으로 복귀, `0.5 <= score < 0.7` 비정형에 남기고 검토 표시, 그 외 비정형 유지.

```bash
python -m marking_pipeline.step4_return \
    --markings markings.json --rescue step3_results.json --out regrouped.json
```
- `markings.json`: `[{"id": "U1", "group": "비정형", "damage": null, "text": "F6"}, ...]`
- `step3_results.json`: `[{"marking_id": "U1", "score": 0.86, "reasons": ["획 두께 일정"]}, ...]`
- 기준값은 `--return-threshold`, `--review-threshold`로 바꿀 수 있습니다.
