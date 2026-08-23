import numpy as np

from thesis_s2s.bargein.detector import DetectorConfig, EnergyVadBaseline
from thesis_s2s.bargein.features import FeatureConfig, frame_feature_matrix
from thesis_s2s.bargein.realtime import PlaybackController
from thesis_s2s.bargein.synthetic import make_clip
from thesis_s2s.bargein.train import train_and_eval
from thesis_s2s.metrics import binary_scores


def test_feature_shapes():
    clip = make_clip("interrupt")
    feats = frame_feature_matrix(clip.audio, FeatureConfig())
    assert feats.ndim == 2
    assert feats.shape[1] == 4 + 13 + 13


def test_energy_vad_and_gbdt_target(tmp_path):
    cfg = DetectorConfig(n_estimators=15, max_depth=2, min_samples_leaf=2)
    report = train_and_eval(n_per_class=8, seed=1, out_dir=tmp_path, detector_config=cfg)
    assert report["proposed"]["n"] >= 4
    assert report["proposed"]["accuracy"] >= 0.80
    assert (tmp_path / "bargein_gbdt.pkl").is_file()


def test_playback_controller_stops():
    class AlwaysInterrupt(EnergyVadBaseline):
        def predict_binary(self, audio, assistant_mask=None):
            return 1

    ctl = PlaybackController(AlwaysInterrupt(), min_consecutive=2)
    ctl.start_playback()
    first = np.ones(1024, dtype=np.float32) * 0.1
    assert ctl.on_mic_chunk(first, interrupt_onset=True) is False
    assert ctl.playing is True
    assert len(ctl._tail) == 1024
    assert ctl.on_mic_chunk(first) is True
    assert ctl.playing is False
    assert ctl.t_barge_in_ms() is not None


def test_baseline_binary_scores_shape():
    y = [0, 1, 0, 1]
    p = [0, 1, 1, 1]
    s = binary_scores(y, p)
    assert s.n == 4
    assert 0 <= s.far <= 1
