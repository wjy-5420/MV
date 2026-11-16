"""Official model APIs + immutable MV v2 input contracts, one model per process."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.remote_training.run import HOME, SOURCES, digest, save

import numpy as np
import torch
from torch.utils.data import DataLoader
from mv_analysis.data.adapters import get_adapter
from mv_analysis.data.team_release_v2 import TeamReleaseV2

NAMES = {'rtdetrv2': 'RT-DETRv2', 'uniad': 'UniAD', 'hvq_trans': 'HVQ-Trans'}


def as_list(samples):
    return samples


def loader(release, model, split, config, category=None):
    task = 'detection' if model == 'rtdetrv2' else 'anomaly'
    ids = None if category is None else release.canonical.loc[release.canonical.category.eq(category), 'image_id'].tolist()
    ds = release.dataset(task=task, split=split, eligible_only=True,
                         adapter=get_adapter(NAMES[model]), image_ids=ids)
    if not len(ds):
        raise ValueError(f'Empty eligible {split} set: {model}/{category}')
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=split == 'train',
                      num_workers=config['workers'], collate_fn=as_list, drop_last=False)


def build(model_name):
    source = HOME / 'sources' / model_name
    if model_name == 'rtdetrv2':
        sys.path.insert(0, str(source / 'rtdetrv2_pytorch'))
        from src.core import YAMLConfig
        cfg = YAMLConfig(str(source / 'rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'),
                         num_classes=8, remap_mscoco_category=False)
        return cfg.model.cuda(), cfg.criterion.cuda(), cfg.postprocessor.cuda()
    sys.path.insert(0, str(source))
    if model_name == 'uniad':
        import yaml
        from models.model_helper import ModelHelper
        from easydict import EasyDict
        from utils.misc_helper import update_config
        cfg = yaml.safe_load((source / 'experiments/MVTec-AD/config.yaml').read_text())
        cfg = update_config(EasyDict(cfg))
        cfg.net[2].kwargs.save_recon = None
        return ModelHelper(cfg.net).cuda(), None, None
    sys.path.insert(0, str(source / 'tools'))
    from models.HVQ_TR_switch import HVQ_TR_switch
    # Official HVQ has 15 category switches. Train one independent model per
    # Real-IAD category with local clslabel=0; never map 30 categories modulo 15.
    return HVQ_TR_switch(channel=272, embed_dim=256).cuda(), None, None


def inputs(samples, name):
    images = torch.stack([s['image'] for s in samples]).cuda()
    if name == 'rtdetrv2':
        targets = []
        for s in samples:
            if not s['detection_trainable']:
                raise ValueError('Invalid D1 supervision reached the model')
            targets.append({'boxes': s['boxes'].cuda(), 'labels': s['labels'].cuda()})
        return images, targets
    return {'image': images, 'clslabel': torch.zeros(len(samples), dtype=torch.long, device='cuda')}, None


def infer(model, post, name, batch):
    output = model(batch)
    if name == 'rtdetrv2':
        # Official postprocessor expects [width,height]; evaluate at resized 640.
        return post(output, torch.full((batch.shape[0], 2), 640, device='cuda'))
    return output['pred']


def anomaly_scores(pred):
    # Same 16x16 average-pooling image score as official evaluator.
    return torch.nn.functional.avg_pool2d(pred, 16, stride=1).flatten(1).max(1).values


def classification_metrics(labels, scores, threshold=None):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
    if len(set(labels)) != 2:
        raise ValueError('AUROC/threshold needs both OK and NG in validation/test')
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        threshold = float(thresholds[int(np.argmax(f1))])
    return {'image_auroc': float(roc_auc_score(labels, scores)),
            'image_f1': float(f1_score(labels, np.asarray(scores) >= threshold)),
            'threshold': threshold, 'images': len(labels)}


@torch.inference_mode()
def evaluate(model, post, name, batches, threshold=None):
    model.eval()
    if name == 'rtdetrv2':
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    labels, scores = [], []
    for samples in batches:
        batch, _ = inputs(samples, name)
        pred = infer(model, post, name, batch)
        if name == 'rtdetrv2':
            metric.update([{k: v.cpu() for k, v in p.items()} for p in pred],
                          [{'boxes': s['boxes_xyxy'], 'labels': s['canonical_labels']} for s in samples])
        else:
            labels.extend(int(s['classification_label']) for s in samples)
            scores.extend(anomaly_scores(pred).cpu().tolist())
    if name == 'rtdetrv2':
        result = metric.compute()
        return {k: float(result[k]) for k in ('map', 'map_50', 'map_75', 'mar_100')}
    return classification_metrics(labels, scores, threshold)


@torch.inference_mode()
def benchmark(model, post, name, sample, config):
    model.eval()
    batch, _ = inputs([sample], name)
    for _ in range(config['benchmark_warmup']):
        out = infer(model, post, name, batch)
        if name != 'rtdetrv2':
            out = anomaly_scores(out)
        del out
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(config['benchmark_repetitions']):
        start = time.perf_counter()
        out = infer(model, post, name, batch)
        if name != 'rtdetrv2':
            out = anomaly_scores(out)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        del out
    return {'batch_size': 1, 'precision': 'FP32', 'fps': 1 / float(np.mean(durations)),
            'latency_ms': 1000 * float(np.mean(durations)),
            'p95_ms': 1000 * float(np.percentile(durations, 95)),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'warmup': config['benchmark_warmup'], 'repetitions': len(durations)}


def train_category(name, category, release, config, folder):
    folder.mkdir(parents=True)
    train = loader(release, name, 'train', config, category)
    val = loader(release, name, 'validation', config, category)
    test = loader(release, name, 'test', config, category)
    if name != 'rtdetrv2':
        for split, batches in [('validation', val), ('test', test)]:
            if set(batches.dataset.data.status) != {'OK', 'NG'}:
                raise ValueError(f'{category}/{split} needs both OK and NG')
    model, criterion, post = build(name)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    best_score = -math.inf
    history = []
    for epoch in range(config['epochs']):
        model.train()
        if name == 'hvq_trans':
            model.enc.eval()
        total, count = 0., 0
        for step, samples in enumerate(train):
            batch, targets = inputs(samples, name)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, targets=targets) if name == 'rtdetrv2' else model(batch)
            if name == 'rtdetrv2':
                loss = sum(criterion(output, targets, epoch=epoch, step=step,
                                     global_step=epoch * len(train) + step).values())
            elif name == 'uniad':
                loss = torch.nn.functional.mse_loss(output['feature_rec'], output['feature_align'])
            else:
                loss = output['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total += float(loss.detach()) * len(samples)
            count += len(samples)
            if step % 20 == 0:
                print(f'{name}/{category} epoch={epoch+1}/50 step={step}/{len(train)} loss={float(loss.detach()):.6f}', flush=True)
        scheduler.step()
        metrics = evaluate(model, post, name, val)
        score = metrics['map'] if name == 'rtdetrv2' else metrics['image_auroc']
        checkpoint = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'config': config, 'category': category,
                      'source_commit': SOURCES[name][1], 'validation': metrics,
                      'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        torch.save(checkpoint, folder / 'last.pt')
        if score > best_score:
            best_score = score
            torch.save(checkpoint, folder / 'best.pt')
        history.append({'epoch': epoch+1, 'loss': total/count, 'validation': metrics})
        save(folder / 'history.json', history)
        print(f'{name}/{category} epoch={epoch+1} validation={metrics}', flush=True)
    # Release optimizer, criterion and training activations before measuring inference.
    del optimizer, scheduler, checkpoint, criterion, output, loss, batch, targets
    model.zero_grad(set_to_none=True)
    ckpt = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    selected_epoch = ckpt['epoch']
    threshold = ckpt['validation'].get('threshold')
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    test_metrics = evaluate(model, post, name, test, threshold)
    performance = benchmark(model, post, name, test.dataset[0], config)
    result = {'category': category or 'ALL_D1', 'best_epoch': selected_epoch,
              'test': test_metrics, 'inference': performance,
              'train_images': len(train.dataset), 'validation_images': len(val.dataset),
              'test_images': len(test.dataset), 'checkpoint_sha256': digest(folder / 'best.pt')}
    save(folder / 'metrics.json', result)
    del model, post
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_report(folder, name, results, error=None):
    lines = [f'# {NAMES[name]}', '', '상태: ' + ('FAILED / 일부 결과만 존재' if error else 'COMPLETED'), '',
             '학습: batch=20, epochs=50, FP32, AdamW lr=1e-4, StepLR(40,0.1), clip=0.1, seed=42.',
             'Validation 최고 지표 checkpoint를 선택하고 Test는 선택 후 한 번 평가합니다.', '',
             '| 품목 | best epoch | Test 성능 | FPS (batch=1) | 평균 ms | P95 ms | allocated GiB | reserved GiB |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        b = r['inference']
        m = ', '.join(f'{k}={v:.4f}' for k, v in r['test'].items() if k != 'images')
        lines.append(f'| {r["category"]} | {r["best_epoch"]} | {m} | {b["fps"]:.2f} | {b["latency_ms"]:.2f} | {b["p95_ms"]:.2f} | {b["peak_allocated_gib"]:.3f} | {b["peak_reserved_gib"]:.3f} |')
    if results:
        keys = ('map', 'map_50', 'map_75') if name == 'rtdetrv2' else ('image_auroc', 'image_f1')
        lines += ['', '품목 단순 평균: ' + ', '.join(f'{k}={np.mean([r["test"][k] for r in results]):.4f}' for k in keys)]
    lines += ['', 'FPS: 파일 I/O, CPU Resize/Normalize, CPU↔GPU 전송 제외. GPU 입력에서 forward+후처리, 10회 warmup + 100회 동기화 측정.',
              '메모리: optimizer/gradient 해제 후 model+입력+forward의 PyTorch peak. CUDA context 등은 제외되므로 필요 VRAM의 하한 참고값입니다.',
              '학습 마지막 batch는 20장 미만일 수 있습니다. OOM이면 batch를 자동 축소하지 않고 실패를 기록합니다.',
              '이상탐지 지표는 영상 AUROC와 Validation 임계값을 고정한 Test F1입니다. 픽셀 AUROC/AUPRO는 이번 실행기에서 산출하지 않습니다.',
              '승인 Crop의 주석 참조 이력을 유지하며 독립 최종 평가로 과장하지 않습니다.',
              '공식 구조를 사용하지만 기존 Real-IAD 입력 계약 및 50에포크 조건으로 변경한 실험입니다. 논문 수치 재현 주장이 아닙니다.',
              f'공식 소스: {SOURCES[name][0]} @ {SOURCES[name][1]}',
              '상세 수치·표본 수·checkpoint SHA256: 각 품목 metrics.json. 실제 환경: ../runtime.json 및 ../pip-freeze.txt.']
    if name == 'hvq_trans':
        lines += ['', 'HVQ-Trans 공식 15개 switch 제약 때문에 품목마다 별도 모델을 학습하고 clslabel=0을 사용합니다. 30품목 단일 통합 모델이 아닙니다.']
    if error:
        lines += ['', '실패 원인: [error.txt](error.txt). 완료되지 않은 품목은 성능값을 생성하지 않았습니다.']
    (folder / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=NAMES, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    settings = json.loads((args.run / 'machine.json').read_text())
    folder = args.run / args.model
    results = []
    try:
        if (config['batch_size'], config['epochs']) != (20, 50):
            raise ValueError('This runner requires batch_size=20, epochs=50')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.benchmark = False
        props = torch.cuda.get_device_properties(0)
        save(args.run / 'runtime.json', {'gpu': props.name, 'vram_gib': props.total_memory/2**30,
                                       'torch': torch.__version__, 'cuda': torch.version.cuda,
                                       'device': torch.cuda.current_device(), 'precision': 'FP32'})
        release = TeamReleaseV2(settings['release'], settings['data_root'])
        categories = [None] if args.model == 'rtdetrv2' else sorted(release.canonical.category.unique())
        for number, category in enumerate(categories):
            # Numeric directory names avoid treating dataset category names as paths.
            result = train_category(args.model, category, release, config, folder / f'category_{number:02d}')
            results.append(result)
            save(folder / 'metrics.json', results)
        write_report(folder, args.model, results)
    except BaseException:
        (folder / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_report(folder, args.model, results, error=True)
        raise


if __name__ == '__main__':
    main()
