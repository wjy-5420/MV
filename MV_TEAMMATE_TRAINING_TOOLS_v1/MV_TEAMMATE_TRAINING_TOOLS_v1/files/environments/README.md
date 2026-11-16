# 모델별 Python 실행 인터페이스

현재 모델별 가상환경은 **설치되지 않았다**. 이 디렉터리는 환경을 분리하는 경로
계약을 정의한다. 기존 공통 `.venv`와 `uv.lock`은 변경하지 않는다.

|모델 키|기본 Python 경로|다른 환경 지정 변수|
|---|---|---|
|rtdetrv2|environments/rtdetrv2/.venv/bin/python|MV_PYTHON_RTDETRV2|
|uniad|environments/uniad/.venv/bin/python|MV_PYTHON_UNIAD|
|hvq_trans|environments/hvq_trans/.venv/bin/python|MV_PYTHON_HVQ_TRANS|
|efficientad_s|environments/efficientad_s/.venv/bin/python|MV_PYTHON_EFFICIENTAD_S|
|retinanet|environments/retinanet/.venv/bin/python|MV_PYTHON_RETINANET|
|yolo11|environments/yolo11/.venv/bin/python|MV_PYTHON_YOLO11|

예를 들어 이미 준비한 모델 환경의 Python을 연결할 때, MV 루트에서 다음처럼 지정한다.

```bash
export MV_PYTHON_RETINANET="/본인/RetinaNet환경/bin/python"
uv run --offline python scripts/experiments/run_weekend.py --models retinanet --dry-run
```

경로는 실제 실행 가능한 Python이어야 한다. 실행기는 해당 Python을 별도 프로세스로
실행하고 `sys.prefix`, torch/torchvision 등의 버전과 CUDA 상태를 기록한다. 경로가
없거나 torch import가 실패하면 BLOCKED다. 현재 학습 backend도 미구현이므로
환경만 준비해도 실제 학습은 시작되지 않는다.

가중치는 `MV_WEIGHTS_<모델키 대문자>` 또는 설정의 `weights`로 **로컬 파일**만
지정한다. 선택적 `weights_sha256`와 대조하고 실제 SHA를 기록한다. 자동 다운로드는 없다.

공식 소스는 기존 고정 Commit 스냅샷 또는 설정 `source_dir`의 깨끗한 공식 clone을
확인한다. 소스 스냅샷 일부를 확보했다는 사실이 전체 패키지 설치 성공을 뜻하지 않는다.
RetinaNet의 설치된 torchvision과 고정 소스의 런타임 일치도는 UNVERIFIED다.

다음 세션에서 모델별 Python/PyTorch/CUDA 조합을 실제 확인한 후 설치 절차를 추가한다.
UniAD의 옛 의존성은 Python 3.12에 그대로 설치할 수 없고, 코드에 직접 `.cuda()`를
호출하는 부분도 있다. HVQ-Trans의 라이선스 확인도 남아 있다. 호환되지 않는
환경을 강제로 공통 `.venv`에 설치하지 않는다.

[실행기 사용법](../docs/experiments/TRAINING_RUNNER_GUIDE.md) ·
[현재 진행 상태](../docs/experiments/TRAINING_IMPLEMENTATION_PROGRESS.md)
