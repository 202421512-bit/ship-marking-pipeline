 # PAC2 — 수기 특징 기반 마킹 추출 시스템 (120장 최종 설계)

## 0. 개발 환경 및 기본 규칙

프로젝트 경로: `C:\AI Study\PAC2`

Python: 3.11.9

가상환경: `.venv`

이미 PyTorch, OpenCV, scikit-learn, scikit-image 등 주요 라이브러리 설치와 import 검증을 마쳤다.

기존 가상환경을 삭제하거나 불필요하게 다시 생성하지 마라. `Source` 폴더의 원본 이미지도 이동·수정·삭제하지 마라.

**최종 목적: 선박 강판 이미지에서 수기로 작성된 문자·숫자·기호의 획을 찾아 추출하고 그 외 영역은 제거한다.**

OCR, 문자 의미 파악, 문자·숫자·기호의 개별 클래스 분류는 구현하지 않는다.

녹과 스크래치를 각각 별도 클래스로 학습하는 것이 아니라, 수기 표기에 대한 형상 특징으로 수기 여부를 판단하는 접근을 우선 적용한다.

## 1. 확정된 데이터셋

경로:

- `Source/formal`: 정형 표기 20장, label=0
- `Source/handwritten`: 수기 표기 100장, label=1

총 120장이다.

기존 프롬프트의 formal 100장, 총 200장이라는 가정은 모두 폐기한다.

우선 실제 이미지 파일 개수, 크기, 확장자, 손상 여부, 중복 이미지, 이미지 내 글자와 배경의 존재 여부를 검사한다.

formal의 의미가 단정한 수기 표기인지 비수기 정형 표기인지도 검사 결과와 사용자에게 받은 정의에 따라 명시한다. 형상 분류의 label=0과 최종 추출에서 제외할 대상의 정의가 일치하는지 확인한다.

데이터 분할:

- 개발 데이터 90장: handwritten 75, formal 15
- 최종 테스트 30장: handwritten 25, formal 5

개발 데이터에 Stratified 5-Fold Cross Validation을 적용한다.

동일 장면에서 생성된 이미지나 증강된 이미지가 학습과 테스트에 동시에 포함되지 않도록 한다. 원본 간 그룹 관계가 있다면 그룹 기반 분할을 우선한다.

Test 30장은 최종 모델 확정 전 사용하지 않는다.

## 2. 전처리와 수기 획 후보 생성

원본 이미지에서 수기 획 후보를 추출하는 OpenCV 전처리 시스템을 구현한다.

CLAHE, Adaptive Threshold, 색상/명도 기반 분리, Morphology, Connected Components 등을 비교한다.

특정 색의 분필이나 펜만 있다고 가정하지 않는다.

전처리 과정에서 수기 후보를 누락하지 않는 방향으로 여러 후보를 생성하고, 후속 형상 특징 분석에서 적합도를 판단한다.

후보가 실제 문자 획인지 아직 알 수 없으므로, 후보 추출 단계의 결과를 정답으로 간주하지 않는다.

전처리 방식과 후보 생성 결과를 이미지로 저장한다.

## 3. PDF 기반 7개 특징 구현

사용자가 제공한 `수기문자_정형비정형_특징설계안.pdf`의 확정된 7개 특징을 적용한다.

1. 획 두께 변화: CV_w
2. 글자 높이 편차: CV_h
4. 문자 간격 편차: CV_g
5. 기울기 변화: S_theta
6. Baseline 흔들림: B
10. 연결요소 크기 분산: CV_A
11. 외곽선 불규칙도: R

특징별 계산식과 초기 임계값은 PDF에 기재된 내용을 따른다.

문자 간격, 높이 편차, Baseline 흔들림은 문자별 Bounding Box가 확보되어야 계산할 수 있다. 연결요소가 반드시 문자 하나를 의미하지 않으므로, 분리 실패 가능성을 처리한다.

특징을 계산할 수 없는 샘플은 결측 여부를 저장하고, 훈련 데이터 기준으로만 결측치 처리 규칙을 학습한다.

PDF의 7개 특징은 수기/정형 형상 분석용이며 녹·스크래치 제거 성능을 직접 보장하지 않는다는 점을 명시한다.

## 4. 특징 표준화와 수기 판단 함수

7개 특징 벡터를 다음과 같이 정의한다.

`X = [CV_w, CV_h, CV_g, S_theta, B, CV_A, R]`

PDF에서 제안한 정형 데이터 기준 표준화를 우선 실험한다.

`z_j = (x_j - mu_formal,j) / (sigma_formal,j + eps)`

평균과 표준편차는 반드시 해당 학습 Fold에서만 계산한다.

이후 PyTorch를 사용하여 Logistic Regression과 동등한 선형 Sigmoid 모델을 구현한다.

`p_handwritten = sigmoid(b + sum(w_j*z_j))`

가중치 w_j와 절편 b를 학습한다.

모델의 손실함수는 클래스 가중 Binary Cross Entropy와 L2 정규화를 사용한다.

`L = WeightedBCE + lambda*sum(w_j^2)`

초기 클래스별 손실 가중치:

- handwritten = 0.6
- formal = 3.0

실제 Fold 클래스 빈도를 기준으로 가중치를 재계산한다. 중복된 클래스 보정을 적용하지 않는다.

Adam 또는 LBFGS 최적화를 비교하고 학습 과정에서 Loss, Gradient, Weight 변화량을 기록한다.

형상 분류의 Loss가 줄어들었다고 이미지 분할 성능이 향상되었다고 해석하지 않는다.

## 5. 추가 모델 비교

기본 Logistic Regression을 먼저 학습한 뒤 다음을 비교한다.

- Random Forest: 7개 특징의 비선형 상호작용 비교
- AdaBoostClassifier: 약한 분류기의 반복 가중치 개선
- GaussianProcessClassifier: 작은 데이터에서 확률 예측과 모델 불확실성 분석

추가로 PyTorch 모델의 학습 샘플에 Hard-example Reweighting을 적용할 수 있도록 한다.

각 기법을 모두 무조건 결합하지 말고 Cross Validation에서 이득이 확인된 모델만 채택한다.

AdaBoostClassifier와 PyTorch 샘플 가중치 재학습은 별도 구현으로 구분한다.

Gaussian Process에서는 분류 확률과 불확실성을 확인하며, R²는 연속형 회귀 문제의 평가 지표이므로 본 이진 분류의 필수 지표로 사용하지 않는다.

## 6. 수기 적합도 기반 최종 추출

테스트 강판 이미지에 대해 다음 과정을 수행한다.

1. 입력 이미지 전처리
2. 수기 획 후보 추출 및 그룹화
3. 후보 영역의 7개 특징 계산
4. 학습한 모델로 수기 확률 추정
5. Validation에서 결정한 임계값으로 후보 보존 또는 제외
6. 보존한 후보의 전경 픽셀로 추출 마스크 생성
7. 원본과 마스크를 결합한 이미지 출력

최종 출력 파일:

- `binary_mask.png`
- `handwritten_only.png`
- `overlay.png`
- `candidate_debug.png`
- `prediction_scores.csv`

후보 영역 전체의 Bounding Box를 흰색으로 채우지 않는다. 전처리에서 추정한 실제 획 픽셀만 마스크로 보존한다.

이 방식의 마스크는 후보 생성 정확도에 영향을 받으므로, 정확한 픽셀 추출 성능을 주장하려면 실제 수기 획의 Ground Truth Mask가 필요하다.

## 7. 검증 및 성능 평가

5-Fold Cross Validation을 진행하고 각 Fold의 예측을 저장한다.

형상 분류 평가 지표:

- Balanced Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC 및 PR-AUC
- Log Loss 및 Brier Score

임계값은 개발 데이터의 교차검증 결과로 결정한다. 미리 정해진 0.5가 무조건 최적이라고 가정하지 않는다.

최종 Test 데이터 30장에 대해 전체 성능과 formal/handwritten 클래스별 성능을 따로 보고한다.

특히 formal 테스트 샘플은 5장뿐이므로 결과의 통계적 한계도 명시한다.

실제 강판 사진에 수기 획 마스크가 존재한다면 추출 Precision, Recall, Dice, IoU를 계산한다.

마스크가 없다면 시각적 추출 결과만 제공하고 픽셀 정확도를 임의로 계산하지 않는다.

## 8. 구현 순서

Phase 1: 기존 가상환경 검사 및 Source 데이터 120장 감사

Phase 2: 전처리, 후보 생성, 7개 특징 추출 함수 구현

Phase 3: 90/30 데이터 분할 및 Stratified 5-Fold 학습

Phase 4: PyTorch Logistic Loss 최적화 및 가중치 분석

Phase 5: Random Forest, AdaBoost, GP 비교

Phase 6: 실제 강판 이미지에서 수기 후보 추출

Phase 7: 최종 Binary Mask 생성 및 오류 분석

Phase 8: Test 30장 평가 및 발표용 결과 저장

각 Phase에서 실제 코드를 작성하고 테스트를 실행하여 결과를 확인한다. 실행하지 않은 학습이나 평가를 성공했다고 보고하지 않는다.

**지금은 Phase 1부터 시작하라.**

먼저 `Source/formal` 20장과 `Source/handwritten` 100장을 실제로 확인하고, 이미지별 특징 계산이 가능한 데이터인지 검사하라.

그다음 데이터 분석 결과를 보여주고 Phase 2의 특징 추출 코드를 구현하라.
