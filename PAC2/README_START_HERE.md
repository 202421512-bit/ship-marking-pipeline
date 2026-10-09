# PAC2 새 프로젝트 시작 방법 (Windows + VS Code)

## 폴더 구조

압축파일에 있는 3개 필수 파일을 **기존** `AI Study/PAC2` 폴더에 넣습니다.
기존 이미지가 있는 `Source/`는 절대 덮어쓰지 않습니다.

```text
AI Study/PAC2/
  Source/
    formal/          # 정형 100장
    handwritten/     # 수기 100장
  setup_pac2.bat
  requirements.txt
  PAC2_VSCODE_PROMPT.md
```

## 실행

1. VS Code 메뉴 `File > Open Folder`에서 **PAC2 폴더**를 엽니다.
2. `Terminal > New Terminal`을 열고 `./setup_pac2.bat` 대신 Windows PowerShell에서는 **`.\setup_pac2.bat`** 를 실행합니다.
3. Python 3.11이 없으면 먼저 설치해야 합니다.
4. VS Code에서 Python 인터프리터를 `.venv/Scripts/python.exe`로 선택합니다.
5. `PAC2_VSCODE_PROMPT.md`를 AI 코딩 에이전트에게 복사하여 전달합니다.
6. 에이전트가 PHASE 0~5 코드를 생성하고 실행하게 합니다.

## 중요한 주의

- 학습 이미지 200장이 실제로 존재하는지는 설치 스크립트가 보장하지 않습니다. PHASE 0 데이터 감사에서 확인합니다.
- `formal=0`, `handwritten=1`이 맞으려면 formal이 실제로 수기 표기가 아닌 비교 대상이어야 합니다. formal도 사람이 쓴 글씨면 라벨 의미부터 수정해야 합니다.
- 정답 Binary Mask가 없다면 픽셀 분할 성능은 측정할 수 없습니다. 출력된 마스크는 7개 특징과 OpenCV 후보 검출의 베이스라인 결과입니다.
- CUDA가 지원되는 GPU를 쓸 때는 시스템의 드라이버/호환 버전을 확인한 후 공식 PyTorch 안내(https://pytorch.org/get-started/locally/)에 맞게 GPU 버전 설치를 검토합니다. 기본 설치에서는 GPU를 가정하지 않습니다.
