"""Optional CPU architecture checks against clean, pinned official checkouts.

MV_TRAINING_TEST_SOURCES=/path/containing/rtdetrv2,uniad,hvq_trans python -m unittest discover -s tests -p test_remote_training_models.py
These use random weights and synthetic inputs; they do not verify CUDA or real data.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
SOURCES = os.environ.get('MV_TRAINING_TEST_SOURCES')


@unittest.skipUnless(HAS_TORCH, 'ML environment required')
class ThresholdTests(unittest.TestCase):
    def test_test_uses_validation_threshold_without_refitting(self):
        from scripts.remote_training.worker import classification_metrics
        val = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        test = classification_metrics([0, 1], [0.7, 0.75], val['threshold'])
        self.assertEqual(test['threshold'], val['threshold'])
        self.assertEqual(test['image_f1'], 0.0)
        self.assertEqual(test['image_auroc'], 1.0)

    def test_single_class_cannot_produce_auroc(self):
        from scripts.remote_training.worker import classification_metrics
        with self.assertRaises(ValueError):
            classification_metrics([0, 0], [0.1, 0.2])


@unittest.skipUnless(HAS_TORCH and SOURCES, 'Set MV_TRAINING_TEST_SOURCES for official CPU smoke')
class OfficialModelsTests(unittest.TestCase):
    def test_all_official_models_forward_backward_eval(self):
        for name in ('rtdetrv2', 'uniad', 'hvq_trans'):
            with self.subTest(model=name):
                subprocess.run([sys.executable, '-c', SMOKE, name, str(ROOT), SOURCES], check=True, timeout=180)


SMOKE = r'''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[2])
from scripts.remote_training import worker as w
import torch
name=sys.argv[1]
source=Path(sys.argv[3])/name
# HOME/sources resolves to the explicitly provided checkouts.
w.HOME=source.parent.parent
assert source == w.HOME/'sources'/name, 'Use a directory named sources'
torch.set_num_threads(2)
if name=='rtdetrv2':
    sys.path.insert(0,str(source/'rtdetrv2_pytorch'))
    from src.core import YAMLConfig
    original=YAMLConfig.__init__
    def init(self,*args,**kwargs):
        kwargs['PResNet']={'pretrained':False}
        original(self,*args,**kwargs)
    setup=patch.object(YAMLConfig,'__init__',init)
else:
    sys.path.insert(0,str(source))
    from models.backbones.efficientnet import model as efficient_model
    setup=patch.object(efficient_model,'load_pretrained_weights',lambda *a,**k:None)
with setup, patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self.cpu()), patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
    model,criterion,post=w.build(name)
    size=640 if name=='rtdetrv2' else 224
    x=torch.randn(1,3,size,size)
    model.train()
    batch={'image':x,'clslabel':torch.zeros(1,dtype=torch.long)}
    if name=='rtdetrv2':
        targets=[{'boxes':torch.tensor([[.5,.5,.1,.1]]),'labels':torch.tensor([0])}]
        output=model(x,targets=targets)
        loss=sum(criterion(output,targets).values())
    else:
        output=model(batch)
        loss=output['loss'] if name=='hvq_trans' else torch.nn.functional.mse_loss(output['feature_rec'],output['feature_align'])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        output=model(x) if name=='rtdetrv2' else model(batch)
        if name=='rtdetrv2':
            pred=post(output,torch.tensor([[640,640]]))
            assert pred[0]['boxes'].shape==(300,4)
        else:
            assert output['pred'].shape==(1,1,224,224)
    print(name,'CPU official architecture forward/backward/eval PASSED (random weights)')
'''

if __name__ == '__main__':
    unittest.main()
