"""Portable survey and failure reporting use no ML dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('remote_run', Path(__file__).resolve().parents[1] / 'scripts/remote_training/run.py')
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class RemoteTrainingTests(unittest.TestCase):
    def test_requested_settings(self):
        c = json.loads((run.ROOT / 'configs/remote_training/default.json').read_text())
        self.assertEqual((c['batch_size'], c['epochs']), (20, 50))
        self.assertEqual(set(c['models']), set(run.SOURCES))

    def test_survey_persists_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, release = root / 'data with spaces', root / 'release'
            data.mkdir()
            release.mkdir()
            (release / 'release.json').write_text('{"release_id":"MV_TEAM_DATA_RELEASE_v2"}')
            args = SimpleNamespace(data_root=str(data), release=str(release), torch_index='test-index')
            with patch.object(run, 'HOME', root / 'out'), patch.object(run, 'capture', return_value={'returncode': 0}):
                run.survey(args)
                settings = json.loads((run.HOME / 'machine.json').read_text())
                self.assertEqual(settings['data_root'], str(data))
                self.assertEqual(settings['release_sha256'], run.digest(release / 'release.json'))
                self.assertTrue((run.HOME / 'environment.md').is_file())

    def test_wrong_release_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'release.json').write_text('{"release_id":"v1"}')
            with self.assertRaises(ValueError):
                run.survey(SimpleNamespace(data_root=tmp, release=tmp))

    def test_setup_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'release.json').write_text('{}')
            with patch.object(run, 'HOME', root / 'out'):
                run.save(run.HOME / 'machine.json', {'release': str(root), 'release_sha256': run.digest(root / 'release.json')})
                with patch.object(run, 'prepare', side_effect=RuntimeError('no CUDA')):
                    with self.assertRaisesRegex(RuntimeError, 'no CUDA'):
                        run.train(SimpleNamespace())
                reports = list(run.HOME.glob('run_*/report.md'))
                self.assertEqual(len(reports), 1)
                self.assertIn('학습 준비 실패', reports[0].read_text())
                self.assertTrue(reports[0].with_name('setup_error.txt').exists())


if __name__ == '__main__':
    unittest.main()
