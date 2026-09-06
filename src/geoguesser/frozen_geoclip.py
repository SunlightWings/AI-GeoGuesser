from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


EARTH_RADIUS_KM = 6371.0088


def normalize_rows(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return value / np.clip(np.linalg.norm(value, axis=1, keepdims=True), eps, None)


def latlng_to_xyz(latlng: np.ndarray) -> np.ndarray:
    radians = np.radians(np.asarray(latlng, dtype=np.float64))
    latitude, longitude = radians[:, 0], radians[:, 1]
    return np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )


def haversine_km(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = latlng_to_xyz(first).astype(np.float64)
    b = latlng_to_xyz(second).astype(np.float64)
    return EARTH_RADIUS_KM * np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0))


def metric_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    distance = haversine_km(truth, prediction)
    return {
        "median_km": float(np.median(distance)),
        "mean_km": float(np.mean(distance)),
        "lt_200_km": float(np.mean(distance < 200.0)),
        "lt_750_km": float(np.mean(distance < 750.0)),
        "p90_km": float(np.quantile(distance, 0.90)),
    }


@dataclass(frozen=True)
class FeatureEncoding:
    train: np.ndarray
    validation: np.ndarray
    description: str


def _standardize(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    return (
        ((train - mean) / std).astype(np.float32),
        ((validation - mean) / std).astype(np.float32),
    )


def spherical_xyz_features(train: np.ndarray, validation: np.ndarray) -> FeatureEncoding:
    first, second = _standardize(latlng_to_xyz(train), latlng_to_xyz(validation))
    return FeatureEncoding(first, second, "3-D unit-sphere XYZ")


def multiscale_rff_features(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
    feature_count: int = 64,
) -> FeatureEncoding:
    if feature_count <= 0 or feature_count % 2:
        raise ValueError("RFF feature_count must be a positive even number")
    rng = np.random.default_rng(seed)
    reference = np.radians(np.mean(train, axis=0))

    def local_km(latlng: np.ndarray) -> np.ndarray:
        radians = np.radians(np.asarray(latlng, dtype=np.float64))
        north = EARTH_RADIUS_KM * (radians[:, 0] - reference[0])
        east = EARTH_RADIUS_KM * np.cos(reference[0]) * (radians[:, 1] - reference[1])
        return np.column_stack((east, north))

    half = feature_count // 2
    base_scales = np.asarray((50, 100, 200, 400, 800, 1600, 3200, 6400), dtype=np.float64)
    scales = np.resize(base_scales, half)
    rng.shuffle(scales)
    direction = rng.uniform(0.0, 2.0 * np.pi, size=half)
    frequency = (2.0 * np.pi / scales)[:, None] * np.column_stack(
        (np.cos(direction), np.sin(direction))
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=half)

    def encode(latlng: np.ndarray) -> np.ndarray:
        projection = local_km(latlng) @ frequency.T + phase[None, :]
        return np.concatenate((np.sin(projection), np.cos(projection)), axis=1)

    first, second = _standardize(encode(train), encode(validation))
    return FeatureEncoding(
        first,
        second,
        f"{feature_count}-D RFF; 50--6400 km wavelengths; seed {seed}",
    )


def haversine_rbf_features(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
    center_count: int = 64,
) -> FeatureEncoding:
    if center_count <= 0 or center_count > len(train):
        raise ValueError("Invalid RBF center count")
    train_xyz = latlng_to_xyz(train).astype(np.float64)
    validation_xyz = latlng_to_xyz(validation).astype(np.float64)
    clustering = KMeans(n_clusters=center_count, n_init=20, random_state=seed).fit(train_xyz)
    centers = clustering.cluster_centers_
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)

    train_distance = EARTH_RADIUS_KM * np.arccos(np.clip(train_xyz @ centers.T, -1.0, 1.0))
    validation_distance = EARTH_RADIUS_KM * np.arccos(
        np.clip(validation_xyz @ centers.T, -1.0, 1.0)
    )
    bandwidth = np.empty(center_count, dtype=np.float64)
    for index in range(center_count):
        member_distance = train_distance[clustering.labels_ == index, index]
        bandwidth[index] = np.median(member_distance) if len(member_distance) else 250.0
    bandwidth = np.clip(1.5 * bandwidth, 80.0, 800.0)
    first = np.exp(-0.5 * (train_distance / bandwidth[None, :]) ** 2)
    second = np.exp(-0.5 * (validation_distance / bandwidth[None, :]) ** 2)
    first, second = _standardize(first, second)
    return FeatureEncoding(first, second, f"{center_count}-center Haversine RBF; seed {seed}")


class TinyGpsTower:
    """Small NumPy MLP trained with symmetric image/GPS InfoNCE."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(
                0.0, np.sqrt(2.0 / (input_dim + hidden_dim)), (input_dim, hidden_dim)
            ).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "w2": rng.normal(
                0.0, np.sqrt(2.0 / (hidden_dim + output_dim)), (hidden_dim, output_dim)
            ).astype(np.float32),
            "b2": np.zeros(output_dim, dtype=np.float32),
        }
        self.moment1 = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.moment2 = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.step = 0

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.tanh(features @ self.parameters["w1"] + self.parameters["b1"])
        raw = hidden @ self.parameters["w2"] + self.parameters["b2"]
        return hidden, raw, normalize_rows(raw)

    def train_batch(
        self,
        features: np.ndarray,
        image_embedding: np.ndarray,
        *,
        temperature: float,
        learning_rate: float,
        weight_decay: float,
    ) -> float:
        hidden, raw, gps_embedding = self.forward(features)
        image_embedding = normalize_rows(image_embedding)
        logits = image_embedding @ gps_embedding.T / float(temperature)
        identity = np.eye(len(logits), dtype=np.float32)

        row_shift = logits - logits.max(axis=1, keepdims=True)
        row_probability = np.exp(row_shift)
        row_probability /= row_probability.sum(axis=1, keepdims=True)
        column_shift = logits - logits.max(axis=0, keepdims=True)
        column_probability = np.exp(column_shift)
        column_probability /= column_probability.sum(axis=0, keepdims=True)
        diagonal = np.arange(len(logits))
        row_loss = -np.log(np.clip(row_probability[diagonal, diagonal], 1e-12, None)).mean()
        column_loss = -np.log(
            np.clip(column_probability[diagonal, diagonal], 1e-12, None)
        ).mean()
        loss = 0.5 * (row_loss + column_loss)

        grad_logits = 0.5 * (
            (row_probability - identity) / len(logits)
            + (column_probability - identity) / len(logits)
        )
        grad_gps = grad_logits.T @ image_embedding / float(temperature)
        norm = np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8, None)
        grad_raw = (
            grad_gps - gps_embedding * (grad_gps * gps_embedding).sum(axis=1, keepdims=True)
        ) / norm
        gradients = {
            "w2": hidden.T @ grad_raw + weight_decay * self.parameters["w2"],
            "b2": grad_raw.sum(axis=0),
        }
        grad_hidden = grad_raw @ self.parameters["w2"].T
        grad_hidden *= 1.0 - hidden**2
        gradients["w1"] = features.T @ grad_hidden + weight_decay * self.parameters["w1"]
        gradients["b1"] = grad_hidden.sum(axis=0)

        self.step += 1
        beta1, beta2 = 0.9, 0.999
        for key, gradient in gradients.items():
            gradient = np.clip(gradient, -5.0, 5.0).astype(np.float32)
            self.moment1[key] = beta1 * self.moment1[key] + (1.0 - beta1) * gradient
            self.moment2[key] = beta2 * self.moment2[key] + (1.0 - beta2) * gradient**2
            corrected1 = self.moment1[key] / (1.0 - beta1**self.step)
            corrected2 = self.moment2[key] / (1.0 - beta2**self.step)
            self.parameters[key] -= learning_rate * corrected1 / (np.sqrt(corrected2) + 1e-8)
        return float(loss)


def train_tower(
    tower: TinyGpsTower,
    features: np.ndarray,
    image_embedding: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(features))
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            if len(index) < 2:
                continue
            losses.append(
                tower.train_batch(
                    features[index],
                    image_embedding[index],
                    temperature=temperature,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
            )
        history.append(float(np.mean(losses)))
        print(f"tower epoch {epoch:03d}: loss={history[-1]:.5f}")
    return history


def nearest_candidate(
    query_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    candidate_latlng: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    query = normalize_rows(query_embedding)
    candidate = normalize_rows(candidate_embedding)
    prediction = []
    for start in range(0, len(query), chunk_size):
        similarity = query[start : start + chunk_size] @ candidate.T
        prediction.append(candidate_latlng[np.argmax(similarity, axis=1)])
    return np.concatenate(prediction, axis=0)
