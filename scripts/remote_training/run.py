"""Two-command remote GPU training entry point; survey uses only the standard library."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / 'outputs' / 'remote_training'
SOURCES = {
    'rtdetrv2': ('https://github.com/lyuwenyu/RT-DETR.git', 'df2a96c8c3a098567a1149bf0248da6efe6d495d'),
    'uniad': ('https://github.com/zhiyuanyou/UniAD.git', 'a5b9d9d20c2aae680fa00fe5088616b611e05f3c'),
    'hvq_trans': ('https://github.com/RuiyingLu/HVQ-Trans.git', '12fbfa96b906d1462cbfc1787e47e2fc17468134'),
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def command(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, cwd=ROOT, **kwargs)


def capture(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def survey(args):
    if not args.data_root or not args.release:
        raise ValueError('survey requires --data-root (original Real-IAD) and --release (extracted v2 directory)')
    data = Path(args.data_root).expanduser().resolve()
    release = Path(args.release).expanduser().resolve()
    if not data.is_dir() or not (release / 'release.json').is_file():
        raise ValueError('Original data directory or extracted release.json does not exist')
    if json.loads((release / 'release.json').read_text())['release_id'] != 'MV_TEAM_DATA_RELEASE_v2':
        raise ValueError('A v2 data release is required')
    settings = {'data_root': str(data), 'release': str(release),
                'release_sha256': digest(release / 'release.json'),
                'torch_index': args.torch_index}
    save(HOME / 'machine.json', settings)
    info = {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
            'architecture': platform.machine(), 'python': sys.version, 'cpu_count': os.cpu_count(),
            'disk_free_gib': shutil.disk_usage(ROOT).free / 2**30,
            'nvidia_smi': capture(['nvidia-smi']),
            'gpu': capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv']),
            'ram': capture(['free', '-h']), 'cpu': capture(['lscpu']), 'git': capture(['git', 'rev-parse', 'HEAD']),
            'tools': {x: shutil.which(x) for x in ('git', 'uv', 'python3.12', 'python3.11')}}
    save(HOME / 'environment.json', info)
    (HOME / 'environment.md').write_text('# 환경 조사\n\n```json\n' + json.dumps(info, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    print(f'환경 조사 저장: {HOME / "environment.md"}\n데이터 경로 저장 완료. 다음: python3 scripts/remote_training/run.py train')


def prepare(settings):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Ubuntu x86_64 / Windows WSL2 Ubuntu에서 실행하세요.')
    env = HOME / 'venv'
    python = env / 'bin/python'
    marker = env / 'mv-ready.json'
    wanted = {'requirements_sha256': digest(Path(__file__).with_name('requirements.txt')),
              'torch': '2.5.1', 'torchvision': '0.20.1', 'index': settings['torch_index']}
    if not marker.exists() or json.loads(marker.read_text()) != wanted:
        if not python.exists():
            uv = shutil.which('uv')
            if uv:
                command([uv, 'venv', '--python', '3.12', '--seed', env])
            else:
                interpreter = shutil.which('python3.12') or shutil.which('python3.11')
                if not interpreter:
                    raise RuntimeError('uv 또는 Python 3.12/3.11 + venv를 먼저 설치하세요. docs/REMOTE_TRAINING.md 참고')
                command([interpreter, '-m', 'venv', env])
        command([python, '-m', 'pip', 'install', 'torch==2.5.1', 'torchvision==0.20.1', '--index-url', settings['torch_index']])
        command([python, '-m', 'pip', 'install', '-r', Path(__file__).with_name('requirements.txt')])
        save(marker, wanted)
    command([python, '-c', 'import torch; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(), "CUDA GPU unavailable"'])
    for name, (url, commit) in SOURCES.items():
        dest = HOME / 'sources' / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            command(['git', 'clone', url, dest])
            command(['git', '-C', dest, 'checkout', '--detach', commit])
        actual = command(['git', '-C', dest, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        if actual != commit:
            raise RuntimeError(f'{name}: source commit mismatch: {actual}')
        dirty = command(['git', '-C', dest, 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout
        if dirty.strip():
            raise RuntimeError(f'{name}: modified official source; use a clean source checkout')
    return python


def report(run, results, config):
    lines = ['# 세 모델 학습 결과', '', f'학습 batch_size={config["batch_size"]}, epochs={config["epochs"]}; 단일 GPU, FP32.', '',
             '| 모델 | 상태 | 결과 |', '|---|---|---|']
    for name, item in results.items():
        lines.append(f'| {name} | {item["status"]} | [상세]({name}/report.md) / [로그]({name}/worker.log) |')
    lines += ['', '| 모델 | Test 성능 (품목 평균) | 평균 FPS | 최대 allocated GiB | 최대 reserved GiB |',
              '|---|---|---:|---:|---:|']
    for name in results:
        metrics_file = run / name / 'metrics.json'
        if not metrics_file.exists():
            continue
        entries = json.loads(metrics_file.read_text())
        if not entries:
            continue
        keys = ('map', 'map_50') if name == 'rtdetrv2' else ('image_auroc', 'image_f1')
        mean = lambda values: sum(values) / len(values)
        scores = ', '.join(f'{key}={mean([x["test"][key] for x in entries]):.4f}' for key in keys)
        fps = mean([x['inference']['fps'] for x in entries])
        allocated = max(x['inference']['peak_allocated_gib'] for x in entries)
        reserved = max(x['inference']['peak_reserved_gib'] for x in entries)
        lines.append(f'| {name} | {scores} | {fps:.2f} | {allocated:.3f} | {reserved:.3f} |')
    lines += ['', '실패 모델에 일부 수치가 있으면 완료된 품목만의 평균입니다. 상태 및 상세 보고서를 함께 확인하세요.',
              '이상탐지는 품목별 모델을 각각 50에포크 학습합니다. Batch는 단일 시점 이미지 20장(마지막 batch는 잔여 수)입니다.',
              'FPS는 batch=1, GPU 상주 입력, warmup 이후 forward+모델 후처리 속도입니다. 파일 읽기·CPU 전처리·전송은 제외합니다.',
              'GPU 메모리는 추론 시 PyTorch peak allocated/reserved GiB이며 GPU 최소 사양의 보증값이 아닙니다. CUDA context와 타 프로세스 메모리는 별도입니다.',
              'RT-DETRv2 mAP와 이상탐지 AUROC/F1은 서로 다른 과제의 지표입니다.',
              '승인 Crop에 기존 주석 참조 이력이 있어 이상탐지 Test의 완전한 독립성을 보장하지 않습니다.', '']
    (run / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def train(args):
    if not (HOME / 'machine.json').is_file():
        raise RuntimeError('먼저 survey --data-root ... --release ... 명령을 실행하세요.')
    settings = json.loads((HOME / 'machine.json').read_text())
    config = json.loads((ROOT / 'configs/remote_training/default.json').read_text())
    run = HOME / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    save(run / 'config.json', config)
    save(run / 'machine.json', settings)
    save(run / 'provenance.json', {'sources': SOURCES,
         'code_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
         'requirements_sha256': digest(Path(__file__).with_name('requirements.txt'))})
    results = {}
    report(run, results, config)
    print(f'결과 폴더: {run}', flush=True)
    try:
        if digest(Path(settings['release']) / 'release.json') != settings['release_sha256']:
            raise RuntimeError('Data release changed since survey; rerun survey')
        python = prepare(settings)
        with (run / 'pip-freeze.txt').open('w') as frozen:
            command([python, '-m', 'pip', 'freeze'], stdout=frozen)
        # Full contract/checksum verification happens before the first model starts.
        command([python, '-c', 'import sys; sys.path.insert(0,sys.argv[1]); from mv_analysis.data.team_release_v2 import TeamReleaseV2; TeamReleaseV2(sys.argv[2],sys.argv[3]); print("DATA VERIFIED")', ROOT, settings['release'], settings['data_root']])
        for name in config['models']:
            folder = run / name
            folder.mkdir()
            print(f'{name}: 학습 시작 (로그 {folder / "worker.log"})', flush=True)
            with (folder / 'worker.log').open('w') as log:
                p = subprocess.run([str(python), '-u', str(Path(__file__).with_name('worker.py')), '--model', name, '--run', str(run)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            results[name] = {'status': 'COMPLETED' if p.returncode == 0 else 'FAILED', 'returncode': p.returncode}
            if not (folder / 'report.md').exists():
                (folder / 'report.md').write_text(f'# {name}\n\nFAILED (exit={p.returncode}). 성능 측정값 없음. [로그](worker.log) 확인.\n', encoding='utf-8')
            save(run / 'state.json', results)
            report(run, results, config)
    except BaseException:
        (run / 'setup_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        (run / 'report.md').write_text('# 학습 준비 실패\n\n학습 완료 아님. [원인](setup_error.txt)을 확인하세요.\n', encoding='utf-8')
        raise
    print(f'최종 보고서: {run / "report.md"}', flush=True)
    if any(x['status'] != 'COMPLETED' for x in results.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('survey')
    p.add_argument('--data-root', default=os.environ.get('REALIAD_DATA_ROOT'))
    p.add_argument('--release', default=os.environ.get('MV_DATA_RELEASE'))
    p.add_argument('--torch-index', default='https://download.pytorch.org/whl/cu124')
    sub.add_parser('train')
    args = parser.parse_args()
    survey(args) if args.action == 'survey' else train(args)


if __name__ == '__main__':
    main()
