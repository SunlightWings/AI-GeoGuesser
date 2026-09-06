import numpy as np
import pandas as pd

from geoguesser.cells import CountryKMeansPartition


def make_frame():
    return pd.DataFrame(
        {
            "country_id": [0] * 4 + [1] * 4,
            "lat": [50.0, 50.2, 51.0, 51.2, 60.0, 60.2, 61.0, 61.2],
            "lng": [10.0, 10.2, 11.0, 11.2, 20.0, 20.2, 21.0, 21.2],
        }
    )


def test_partition_fit_transform_and_state_roundtrip():
    frame = make_frame()
    partition = CountryKMeansPartition(target_size=2, random_state=1, n_init=2)
    labels = partition.fit_transform(frame)
    assert partition.n_cells_ == 4
    assert labels.shape == (8,)
    assert np.all(partition.cell_country_ids_[labels] == frame["country_id"].to_numpy())
    restored = CountryKMeansPartition.from_state(partition.to_state())
    np.testing.assert_array_equal(restored.transform(frame), labels)

