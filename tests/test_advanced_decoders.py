import numpy as np

from geoguesser.advanced_decoders import (
    build_intersection_partition,
    decode_cplanet_numpy,
    decode_fine_numpy,
)
from geoguesser.frozen_geoclip import TinyGpsTower, spherical_xyz_features


def test_cached_fine_and_intersection_decoders_are_finite():
    latlng = np.array([[50.0, 10.0], [50.1, 10.1], [60.0, 20.0], [60.1, 20.1]])
    partition = build_intersection_partition(
        latlng,
        fine_cell=np.array([0, 0, 1, 1]),
        coarse_cell=np.array([0, 1, 2, 2]),
        country_id=np.array([0, 0, 1, 1]),
    )
    output = {
        "fine_logits": np.array([[5.0, 0.0], [0.0, 5.0]]),
        "coarse_logits": np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        "country_logits": np.array([[5.0, 0.0], [0.0, 5.0]]),
        "fine_residuals": np.zeros((2, 2, 3)),
        "global_xyz": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    centers = np.array([[0.633, 0.112, 0.766], [0.470, 0.171, 0.866]])
    fine = decode_fine_numpy(
        output,
        centers,
        np.array([0, 1]),
        top_cells=1,
        temperature=1.0,
        country_gate=1.0,
        residual_scale=0.0,
        global_blend=0.0,
    )
    cplanet = decode_cplanet_numpy(
        output,
        partition,
        top_intersections=1,
        temperature=1.0,
        country_gate=1.0,
        coarse_weight=1.0,
    )
    assert fine.shape == cplanet.shape == (2, 2)
    assert np.isfinite(fine).all() and np.isfinite(cplanet).all()


def test_tiny_gps_tower_updates_with_finite_loss():
    coordinates = np.array([[50.0, 10.0], [51.0, 11.0], [52.0, 12.0]])
    encoding = spherical_xyz_features(coordinates, coordinates[:1])
    tower = TinyGpsTower(encoding.train.shape[1], output_dim=8, hidden_dim=4, seed=1)
    before = tower.parameters["w1"].copy()
    loss = tower.train_batch(
        encoding.train,
        np.eye(3, 8, dtype=np.float32),
        temperature=0.1,
        learning_rate=0.001,
        weight_decay=0.0,
    )
    assert np.isfinite(loss)
    assert not np.array_equal(before, tower.parameters["w1"])
