import numpy as np

from geoguesser.geo import (
    ensemble_latlng,
    haversine_km_np,
    latlng_to_unit_np,
    metric_summary,
    unit_to_latlng_np,
)
from geoguesser.retrieval import RetrievalConfig, hybrid_retrieval_predict


def test_haversine_identity_and_known_scale():
    assert float(haversine_km_np(52.52, 13.405, 52.52, 13.405)) == 0.0
    paris_berlin = float(haversine_km_np(48.8566, 2.3522, 52.52, 13.405))
    assert 870.0 < paris_berlin < 890.0


def test_spherical_round_trip():
    coordinates = np.array([[65.0, -22.0], [36.8, 30.8], [51.0, 8.2]])
    recovered = unit_to_latlng_np(latlng_to_unit_np(coordinates))
    np.testing.assert_allclose(recovered, coordinates, atol=1e-6)


def test_metric_and_ensemble_shapes():
    truth = np.array([[50.0, 10.0], [60.0, 20.0]])
    summary = metric_summary(truth, truth.copy())
    assert summary["median_km"] == 0.0
    prediction = ensemble_latlng([truth, truth])
    np.testing.assert_allclose(prediction, truth, atol=1e-6)


def test_retrieval_exact_neighbor():
    bank_latlng = np.array([[50.0, 10.0], [60.0, 20.0]])
    prediction = hybrid_retrieval_predict(
        query_embedding=np.array([[1.0, 0.0]]),
        query_fine_logits=np.array([[5.0, -5.0]]),
        query_country_logits=np.array([[5.0, -5.0]]),
        base_latlng=np.array([[0.0, 0.0]]),
        bank_embedding=np.array([[1.0, 0.0], [0.0, 1.0]]),
        bank_latlng=bank_latlng,
        bank_fine_cell=np.array([0, 1]),
        cell_country_ids=np.array([0, 1]),
        country_gate=1.0,
        config=RetrievalConfig(top_cells=1, neighbors=1, retrieval_blend=1.0),
    )
    np.testing.assert_allclose(prediction, bank_latlng[:1], atol=1e-5)
