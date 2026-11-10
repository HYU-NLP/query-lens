import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "evaluation/input_score.py"
BOOTSTRAP = r'''
import json, pickle, runpy, sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

class Feature:
    def __init__(self, name):
        self.name = name
        self.activations = [SimpleNamespace(max_value_token_index=0,
                            get_values=lambda: [1], get_tokens=lambda token=token: [token],
                            get_tokens_str=lambda token=token: token) for token in 'abcd']
    def __str__(self):
        return self.name

util = ModuleType('util')
util.Feature = Feature
util.Activation = SimpleNamespace
def read_features(path):
    with open(path, 'rb') as handle:
        return [Feature(name) for name in pickle.load(handle)]
util.read_pkl_file = read_features
util.normalize_tokens = list  # Fixture tokens are plain ASCII without tokenizer markers.
sys.modules['util'] = util

import matplotlib
matplotlib.use('Agg')
from matplotlib.axes import Axes
original = Axes.plot
def record_plot(self, x, y, *args, **kwargs):
    Path('plotted.json').write_text(json.dumps({'layers': list(x), 'means': list(y)}))
    return original(self, x, y, *args, **kwargs)
Axes.plot = record_plot
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


def fixture(directory, names=None):
    rows = [("Feature test/10/0", list("abcd")), ("Feature test/2/0", []),
            ("Feature test/5/0", list("ab")), ("Feature test/2/1", list("ab"))]
    if names is not None:
        rows = [(name, list("ab")) for name in names]
    tokens = directory / "tokens.jsonl"
    tokens.write_text("".join(json.dumps({"feature_str": name, "feature_index": i,
                                         "top_tokens": top}) + "\n"
                               for i, (name, top) in enumerate(rows)), encoding="utf-8")
    features = directory / "features.pkl"
    features.write_bytes(pickle.dumps([name for name, _ in rows]))
    return tokens, features


def run_cli(directory, tokens, features, *extra):
    return subprocess.run([sys.executable, "-c", BOOTSTRAP, str(CLI), "--jsonl", str(tokens),
                           "--features", str(features), *extra], cwd=directory,
                          capture_output=True, text=True, timeout=60)


class InputVisualizationTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)
        self.tokens, self.features = fixture(self.directory)

    def successful_run(self, *extra):
        run = run_cli(self.directory, self.tokens, self.features, *extra)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return run

    def test_visualize_creates_png_with_numeric_layer_means_and_preserves_json(self):
        self.successful_run()
        paths = [self.directory / name for name in ("feature_activation_results.json", "summary.json")]
        before = [path.read_bytes() for path in paths]
        self.assertFalse(list(self.directory.glob("*.png")))
        run = self.successful_run("--visualize")
        image = self.directory / "feature_activation_results.png"
        self.assertTrue(image.is_file(), run.stdout + run.stderr)
        self.assertGreater(image.stat().st_size, 100)
        self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn(str(image), run.stdout)
        self.assertEqual(json.loads((self.directory / "plotted.json").read_text()),
                         {"layers": [2, 5, 10], "means": [0.25, 0.5, 1.0]})
        self.assertEqual([path.read_bytes() for path in paths], before)

    def test_custom_output_filename(self):
        run = self.successful_run("--output", "nested/custom.json", "--visualize")
        image = self.directory / "nested/custom.png"
        self.assertTrue(image.is_file(), run.stdout + run.stderr)
        self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue((image.parent / "summary.json").is_file())

    def test_without_flag_writes_only_scores_and_summary(self):
        self.successful_run()
        self.assertFalse(list(self.directory.rglob("*.png")))
        self.assertFalse((self.directory / "plotted.json").exists())
        payload = json.loads((self.directory / "feature_activation_results.json").read_text())
        self.assertEqual(set(payload), {"features", "summary", "output_path"})
        self.assertEqual(len(payload["features"]), 4)

    def test_empty_or_unparseable_layers_warn_without_plot(self):
        for names in ([], ["Feature without a layer"]):
            with self.subTest(names=names):
                self.tokens, self.features = fixture(self.directory, names)
                run = self.successful_run("--visualize")
                self.assertIn("No usable layer scores", run.stdout + run.stderr)
                self.assertNotIn("Plot saved", run.stdout + run.stderr)
                self.assertFalse(list(self.directory.rglob("*.png")))

    def test_save_errors_propagate(self):
        # A directory at the PNG path produces a real filesystem error in savefig.
        (self.directory / "feature_activation_results.png").mkdir()
        run = run_cli(self.directory, self.tokens, self.features, "--visualize")
        self.assertNotEqual(run.returncode, 0)
        self.assertNotIn("Plot saved", run.stdout)
