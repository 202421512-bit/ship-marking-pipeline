"""pytest 설정: 저장소 맨 위에서 pytest 를 돌려도 이 데모의 테스트가 올바르게 실행·건너뛰기 되도록.

- 이 폴더를 import 경로에 추가 (테스트가 check, hw_demo 를 import 함)
- torch 가 설치되지 않은 환경(예: 팀 CI 는 opencv·numpy·pytest 만 설치)에서는 이 데모의 테스트를 건너뜀
  → 데모 테스트는 이 폴더에서 ./setup.sh 로 설치한 뒤 `.venv/bin/python -m pytest -q` 로 실행
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
    collect_ignore_glob = ["*"]
