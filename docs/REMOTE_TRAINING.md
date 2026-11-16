# 다른 PC에서 세 모델 학습

이번 변경이 포함된 프로젝트 전체를 대상 PC에 준비합니다. 기존 TRAINING_TOOLS v1 ZIP만으로는 학습되지 않습니다. 원본 Real-IAD와 검증·압축 해제한 `MV_TEAM_DATA_RELEASE_v2`도 별도로 준비해야 합니다.

지원: NVIDIA GPU를 사용하는 Ubuntu x86_64 / Windows WSL2 Ubuntu. `git`, `python3`, 인터넷, `uv` 또는 Python 3.12/3.11 + venv가 필요합니다. Windows PowerShell 직접 실행은 지원하지 않습니다. NVIDIA 드라이버는 CUDA 12.4 PyTorch wheel과 호환되어야 합니다. 시스템 CUDA Toolkit 설치는 필수가 아닙니다.

프로젝트 루트에서 다음 두 명령을 실행합니다. 첫 명령의 두 경로만 실제 경로로 바꿉니다. 원본 루트 바로 아래에는 품목 폴더가, release 경로 바로 아래에는 `release.json`이 있어야 합니다.

```bash
python3 scripts/remote_training/run.py survey --data-root "/데이터/Real-IAD" --release "/데이터/MV_TEAM_DATA_RELEASE_v2"
```

```bash
python3 scripts/remote_training/run.py train
```

환경 조사는 OS·CPU·RAM·디스크·GPU·드라이버를 기록하고 데이터 경로를 저장합니다. 학습 명령은 별도 `outputs/remote_training/venv`에 PyTorch 2.5.1 / torchvision 0.20.1 CUDA 환경과 의존성을 설치하고, 고정 커밋의 공식 소스를 다운로드하고, 릴리스 계약/해시를 검사한 다음 RT-DETRv2 → UniAD → HVQ-Trans 순서로 실행합니다. 첫 실행에는 공식 ImageNet backbone 가중치 다운로드도 필요합니다. 프로젝트 공통 CPU `.venv` 및 `uv.lock`은 변경하지 않습니다.

## 학습·평가 조건

- 실제 학습 batch=20, epochs=50, FP32, 단일 GPU, gradient accumulation 없음. 마지막 batch는 잔여 이미지 수입니다. OOM이면 자동으로 batch를 줄이지 않고 오류를 기록합니다.
- RT-DETRv2: 공식 R18-vd, 640×640, 원본 전체 D1 결함 8종. ImageNet backbone 초기화, 탐지 head 신규 학습. Test COCO mAP@[.50:.95], mAP50, mAP75, mAR100.
- UniAD: 공식 EfficientNet-B4 + UniAD, 승인 Crop → 224×224 ImageNet 정규화. 품목별 정상 Train 학습.
- HVQ-Trans: 공식 HVQ_TR_switch(channel=272, embed_dim=256), 승인 Crop → 224×224. 공식 15개 switch 제약 때문에 Real-IAD 30품목은 품목별 별도 모델을 학습하고 로컬 clslabel=0을 사용합니다. 품목을 임의로 합치지 않습니다.
- 이상탐지 두 모델은 **품목마다 각각 50에포크**입니다. 전체 30품목이면 각각 30개 모델을 만들므로 시간이 오래 걸릴 수 있습니다.
- AdamW lr=0.0001, weight_decay=0.0001, StepLR(40,0.1), gradient clip=0.1, seed=42. RT-DETRv2 공식 augmentation/EMA는 적용하지 않습니다. HVQ frozen backbone은 eval 상태를 유지합니다. 논문 수치 재현 조건과는 다른 Real-IAD 실험입니다.
- 매 에포크 Validation AUROC(이상탐지) / mAP(탐지) 최고 checkpoint 선택 후 Test를 한 번 평가합니다. 이상탐지 score는 공식 방식인 16×16 average-pooling 후 최대값이며 Validation F1 최대 임계값을 Test에 고정합니다. Validation/Test에 품목별 OK·NG가 둘 다 필요합니다. 영상 AUROC/F1을 보고하며 픽셀 AUROC/AUPRO는 산출하지 않습니다.
- FPS: batch=1, FP32, GPU 상주 입력, forward+후처리, warmup 10회 + 동기화 100회. I/O·CPU 전처리·전송 제외. 평균/P95 latency도 저장합니다.
- 추론 VRAM: optimizer/gradient 해제 후 model+입력+activation의 PyTorch peak allocated/reserved GiB. CUDA context·타 프로세스는 제외되므로 구매 시 보장되는 최소 VRAM 수치는 아닙니다.
- 승인 Crop의 기존 주석 참조 이력을 보존하며 완전히 독립적인 최종 평가라고 보장하지 않습니다.

## 결과

`outputs/remote_training/environment.md`: 환경 조사.

`outputs/remote_training/run_<UTC>_<ID>/report.md`: 전체 결과와 모델별 보고서 링크.

각 `rtdetrv2/`, `uniad/`, `hvq_trans/` 아래 `report.md`에 성능·FPS·평균/P95 latency·추론 GPU 메모리가 표로 저장됩니다. `metrics.json`, `worker.log`와 각 `category_00/` 등의 `best.pt`, `last.pt`, `history.json`, `metrics.json`도 저장합니다. 품목 이름과 표본 수, best checkpoint SHA256은 metrics.json에서 확인합니다. 실제 설치 버전은 run의 `pip-freeze.txt`, GPU/CUDA/torch는 `runtime.json`에 기록합니다.

준비 실패 시 `setup_error.txt`와 실패 보고서를 생성합니다. 모델 실패 시 그 모델 오류를 남기고 다음 모델을 시도하며 최종 종료 코드는 1입니다. CUDA가 없으면 CPU 학습으로 전환하지 않습니다. 재실행은 새 run이며 **자동 중단점 재개는 지원하지 않습니다**. 가중치와 로컬 데이터 경로가 포함된 outputs는 Git 제외 대상입니다.

`uv`가 없으면 [기존 Linux 설치 안내](INSTALL_LINUX.md)를 따르거나 Python 3.12/3.11 + venv를 준비합니다. 데이터 ZIP 미해제 상태라면 기존 `scripts/team_data_release.py verify`로 전달받은 SHA256을 검증하고 해제한 뒤 실행합니다. 오래된 드라이버는 호환 드라이버를 준비하거나 survey에 `--torch-index https://download.pytorch.org/whl/cu121`을 추가할 수 있습니다. 최신 GPU가 PyTorch 2.5.1 지원 범위를 벗어나는 경우 별도 환경 검증이 필요합니다.

개발 PC에서 50에포크 GPU 학습은 수행하지 않았습니다. 실제 성능/FPS/VRAM은 대상 PC에서 실행 후 생성되는 측정값으로 확인합니다.

## 개발 검증 기록

- 표준 라이브러리 CLI 테스트 4개 통과: batch/epoch 설정, 경로 저장(공백 포함), 잘못된 릴리스 거부, 준비 실패 시 Markdown 생성.
- ML 테스트 3개 통과: 고정한 공식 소스 세 모델의 CPU synthetic forward/backward/eval, Validation 임계값 고정, 단일 클래스 AUROC 거부.
- CPU 구조 검증에서는 backbone 다운로드를 차단하고 무작위 가중치를 사용했습니다. 실제 학습에서는 ImageNet 가중치를 사용합니다. 이 검증은 실제 데이터·batch 20·50에포크·CUDA 학습 성공을 의미하지 않습니다.

CLI 테스트:

```bash
python3 -m unittest discover -s tests -p test_remote_training_cli.py -v
```

공식 소스가 준비된 ML 환경에서 선택적으로 CPU 구조 테스트를 실행할 수 있습니다(환경 설치 후의 venv 경로 사용).

```bash
MV_TRAINING_TEST_SOURCES="$PWD/outputs/remote_training/sources" outputs/remote_training/venv/bin/python -m unittest discover -s tests -p test_remote_training_models.py -v
```
