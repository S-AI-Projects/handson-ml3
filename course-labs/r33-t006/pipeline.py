"""
R33-T006: Bounded CPU Smoke Run for Autoencoder with Tied Weights
Chapter 17: Autoencoders, GANs, and Diffusion Models / Tying Weights

Contract:
- Bounded CPU image pipeline smoke test: <= 32 samples, fixed seed, <= 1 epoch / <= 20 update steps.
- Tied weights: decoder weight matrix is strictly the transpose of the encoder weight matrix (W_dec = W_enc^T).
- Provides both TiedWeightsAutoencoder and an explicitly labelled NaiveConstantModel baseline.
- Offline, CPU-only, standard-library implementation with deterministic reproducibility.
- Labelled strictly as pipeline verification on synthetic data, NOT real-world performance evidence.
"""

from __future__ import annotations
import csv
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# Maximum execution bounds declared in task specification
MAX_ALLOWED_SAMPLES = 32
MAX_ALLOWED_STEPS = 20
MAX_ALLOWED_EPOCHS = 1
IMAGE_HEIGHT = 4
IMAGE_WIDTH = 4
NUM_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH  # 16


@dataclass
class DatasetRecord:
    record_id: str
    group_id: str
    split: str
    pixels: List[int]
    pixels_2d: List[List[int]]
    label: int

    def __post_init__(self):
        if len(self.pixels) != NUM_PIXELS:
            raise ValueError(f"Record {self.record_id}: Expected {NUM_PIXELS} pixels, got {len(self.pixels)}")
        for p in self.pixels:
            if not (0 <= p <= 255):
                raise ValueError(f"Record {self.record_id}: Pixel value {p} outside [0, 255]")
        if self.label not in (0, 1):
            raise ValueError(f"Record {self.record_id}: Expected binary label (0 or 1), got {self.label}")


@dataclass
class SmokeRunResult:
    model_name: str
    is_baseline: bool
    weights_tied: bool
    num_samples: int
    input_shape: Tuple[int, int]
    initial_loss: float
    final_loss: float
    loss_is_finite: bool
    steps_executed: int
    epochs_executed: int
    reconstructions_flat: List[List[float]]
    reconstructions_2d: List[List[List[float]]]
    weights_encoder: List[List[float]] = field(default_factory=list)
    weights_decoder: List[List[float]] = field(default_factory=list)
    cpu_only: bool = True
    synthetic_verification_only: bool = True


def load_dataset(filepath: Optional[str] = None, format: str = "json") -> List[DatasetRecord]:
    """
    Load the synthetic 4x4 grayscale image fixture from JSON or CSV.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "fixture.json" if format.lower() == "json" else "fixture.csv"
        filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fixture file not found: {filepath}")

    records: List[DatasetRecord] = []

    if format.lower() == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["records"]:
            record = DatasetRecord(
                record_id=item["record_id"],
                group_id=item["group_id"],
                split=item.get("split", "train"),
                pixels=list(map(int, item["pixels"])),
                pixels_2d=[list(map(int, row)) for row in item.get("pixels_2d", [])]
                if "pixels_2d" in item
                else [list(map(int, item["pixels"][r * 4 : (r + 1) * 4])) for r in range(4)],
                label=int(item["label"]),
            )
            records.append(record)

    elif format.lower() == "csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_pixels = row["pixels"].replace('"', "").replace(" ", "").split(",")
                pixels = [int(p) for p in raw_pixels]
                pixels_2d = [pixels[r * 4 : (r + 1) * 4] for r in range(4)]
                record = DatasetRecord(
                    record_id=row["record_id"],
                    group_id=row["group_id"],
                    split=row.get("split", "train"),
                    pixels=pixels,
                    pixels_2d=pixels_2d,
                    label=int(row["label"]),
                )
                records.append(record)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    return records


class TiedWeightsAutoencoder:
    """
    Autoencoder with Tied Weights: W_dec = W_enc^T.
    The decoder has NO independent weight matrix. It shares W_enc transposed.
    Both encoder and decoder gradients update W_enc simultaneously.
    """

    def __init__(self, input_dim: int = NUM_PIXELS, latent_dim: int = 4, seed: int = 42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seed = seed
        self.is_baseline = False
        self.weights_tied = True

        # Deterministic parameter initialization
        rng = random.Random(seed)
        limit = math.sqrt(6.0 / (input_dim + latent_dim))
        # W_enc is D x K
        self.W_enc: List[List[float]] = [
            [rng.uniform(-limit, limit) for _ in range(latent_dim)] for _ in range(input_dim)
        ]
        self.b_enc: List[float] = [0.0] * latent_dim
        self.b_dec: List[float] = [0.0] * input_dim

    @property
    def W_dec(self) -> List[List[float]]:
        """
        Decoder weight matrix is identically the transpose of W_enc (Tied Weights Contract).
        Shape: latent_dim x input_dim (K x D).
        """
        return [[self.W_enc[i][k] for i in range(self.input_dim)] for k in range(self.latent_dim)]

    def encode(self, X: List[List[float]]) -> List[List[float]]:
        """
        Z = X * W_enc + b_enc
        X: N x D, W_enc: D x K -> Z: N x K
        """
        N = len(X)
        Z: List[List[float]] = []
        for i in range(N):
            row_z = []
            for k in range(self.latent_dim):
                val = self.b_enc[k] + sum(X[i][j] * self.W_enc[j][k] for j in range(self.input_dim))
                row_z.append(val)
            Z.append(row_z)
        return Z

    def decode(self, Z: List[List[float]]) -> List[List[float]]:
        """
        X_hat = Z * W_dec + b_dec = Z * (W_enc^T) + b_dec
        Z: N x K, W_dec: K x D -> X_hat: N x D
        """
        N = len(Z)
        X_hat: List[List[float]] = []
        for i in range(N):
            row_x = []
            for j in range(self.input_dim):
                val = self.b_dec[j] + sum(Z[i][k] * self.W_enc[j][k] for k in range(self.latent_dim))
                row_x.append(val)
            X_hat.append(row_x)
        return X_hat

    def forward(self, X: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        Z = self.encode(X)
        X_hat = self.decode(Z)
        return Z, X_hat

    @staticmethod
    def compute_mse_loss(X: List[List[float]], X_hat: List[List[float]]) -> float:
        N = len(X)
        D = len(X[0])
        total_sq_err = sum((X[i][j] - X_hat[i][j]) ** 2 for i in range(N) for j in range(D))
        return total_sq_err / (N * D)

    def fit_step(self, X: List[List[float]], lr: float = 0.05) -> float:
        """
        Perform a single batch update step.
        Computes analytical gradients for tied weights:
        dW = dW_from_encoder + dW_from_decoder
        """
        N = len(X)
        D = self.input_dim
        K = self.latent_dim

        # Forward pass
        Z, X_hat = self.forward(X)
        loss = self.compute_mse_loss(X, X_hat)

        # Output error derivative: dLoss/dX_hat = (2 / (N * D)) * (X_hat - X)
        scale = 2.0 / (N * D)
        dX_hat: List[List[float]] = [[scale * (X_hat[i][j] - X[i][j]) for j in range(D)] for i in range(N)]

        # Decoder bias gradient
        db_dec = [sum(dX_hat[i][j] for i in range(N)) for j in range(D)]

        # Decoder gradient to latent Z: dZ = dX_hat * W_enc (N x K)
        dZ: List[List[float]] = []
        for i in range(N):
            row_dz = [sum(dX_hat[i][j] * self.W_enc[j][k] for j in range(D)) for k in range(K)]
            dZ.append(row_dz)

        # Decoder gradient to tied weight W_enc:
        # W_dec is K x D, dW_dec = Z^T * dX_hat (K x D).
        # Since W_dec = W_enc^T, dW_enc_from_dec = (dW_dec)^T = dX_hat^T * Z (D x K).
        dW_dec_contrib = [
            [sum(dX_hat[i][j] * Z[i][k] for i in range(N)) for k in range(K)] for j in range(D)
        ]

        # Encoder bias gradient
        db_enc = [sum(dZ[i][k] for i in range(N)) for k in range(K)]

        # Encoder gradient to W_enc: X^T * dZ (D x K)
        dW_enc_contrib = [
            [sum(X[i][j] * dZ[i][k] for i in range(N)) for k in range(K)] for j in range(D)
        ]

        # Total tied gradient: sum of encoder and decoder contributions
        for j in range(D):
            for k in range(K):
                total_grad = dW_enc_contrib[j][k] + dW_dec_contrib[j][k]
                self.W_enc[j][k] -= lr * total_grad

        for j in range(D):
            self.b_dec[j] -= lr * db_dec[j]

        for k in range(K):
            self.b_enc[k] -= lr * db_enc[k]

        return loss

    def run_smoke_test(
        self,
        records: List[DatasetRecord],
        max_steps: int = 10,
        lr: float = 0.05,
    ) -> SmokeRunResult:
        """
        Execute bounded CPU smoke run on up to 32 samples for <= 20 steps.
        """
        if len(records) > MAX_ALLOWED_SAMPLES:
            raise ValueError(
                f"Sample count {len(records)} exceeds maximum allowed bounded limit ({MAX_ALLOWED_SAMPLES})"
            )
        if max_steps > MAX_ALLOWED_STEPS:
            raise ValueError(
                f"Requested steps {max_steps} exceeds maximum allowed bounded limit ({MAX_ALLOWED_STEPS})"
            )

        # Normalize pixel values to [0.0, 1.0]
        X: List[List[float]] = [[p / 255.0 for p in r.pixels] for r in records]

        # Initial forward pass and loss
        _, X_hat_init = self.forward(X)
        initial_loss = self.compute_mse_loss(X, X_hat_init)

        # Training steps (at most max_steps, within 1 epoch of batch updates)
        steps_executed = 0
        current_loss = initial_loss
        for _ in range(max_steps):
            current_loss = self.fit_step(X, lr=lr)
            steps_executed += 1

        # Final forward pass
        _, X_hat_final = self.forward(X)
        final_loss = self.compute_mse_loss(X, X_hat_final)

        # Check finiteness
        loss_is_finite = math.isfinite(initial_loss) and math.isfinite(final_loss)

        # Scale reconstructions back to [0, 255] and clamp
        reconstructions_flat: List[List[float]] = []
        reconstructions_2d: List[List[List[float]]] = []
        for row in X_hat_final:
            recon_row = [max(0.0, min(255.0, val * 255.0)) for val in row]
            reconstructions_flat.append(recon_row)
            recon_2d = [recon_row[r * 4 : (r + 1) * 4] for r in range(4)]
            reconstructions_2d.append(recon_2d)

        return SmokeRunResult(
            model_name="TiedWeightsAutoencoder",
            is_baseline=False,
            weights_tied=True,
            num_samples=len(records),
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_is_finite=loss_is_finite,
            steps_executed=steps_executed,
            epochs_executed=1,
            reconstructions_flat=reconstructions_flat,
            reconstructions_2d=reconstructions_2d,
            weights_encoder=[row[:] for row in self.W_enc],
            weights_decoder=[row[:] for row in self.W_dec],
            cpu_only=True,
            synthetic_verification_only=True,
        )


class NaiveConstantModel:
    """
    Explicitly labelled deterministic baseline model.
    Predicts constant mean training pixel values across all pixels.
    Used for verifying interface consistency against a naive predictor.
    """

    def __init__(self):
        self.is_baseline = True
        self.weights_tied = False
        self.mean_pixels: List[float] = [0.0] * NUM_PIXELS

    def fit(self, records: List[DatasetRecord]) -> None:
        N = len(records)
        if N == 0:
            return
        sums = [0.0] * NUM_PIXELS
        for r in records:
            for j in range(NUM_PIXELS):
                sums[j] += r.pixels[j] / 255.0
        self.mean_pixels = [s / N for s in sums]

    def run_smoke_test(
        self,
        records: List[DatasetRecord],
        max_steps: int = 1,
    ) -> SmokeRunResult:
        if len(records) > MAX_ALLOWED_SAMPLES:
            raise ValueError(
                f"Sample count {len(records)} exceeds maximum allowed bounded limit ({MAX_ALLOWED_SAMPLES})"
            )
        if max_steps > MAX_ALLOWED_STEPS:
            raise ValueError(
                f"Requested steps {max_steps} exceeds maximum allowed bounded limit ({MAX_ALLOWED_STEPS})"
            )

        X: List[List[float]] = [[p / 255.0 for p in r.pixels] for r in records]
        self.fit(records)

        # Baseline outputs constant mean
        X_hat: List[List[float]] = [self.mean_pixels[:] for _ in records]
        loss = TiedWeightsAutoencoder.compute_mse_loss(X, X_hat)

        reconstructions_flat: List[List[float]] = []
        reconstructions_2d: List[List[List[float]]] = []
        for row in X_hat:
            recon_row = [max(0.0, min(255.0, val * 255.0)) for val in row]
            reconstructions_flat.append(recon_row)
            recon_2d = [recon_row[r * 4 : (r + 1) * 4] for r in range(4)]
            reconstructions_2d.append(recon_2d)

        return SmokeRunResult(
            model_name="NaiveConstantModel",
            is_baseline=True,
            weights_tied=False,
            num_samples=len(records),
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
            initial_loss=loss,
            final_loss=loss,
            loss_is_finite=math.isfinite(loss),
            steps_executed=1,
            epochs_executed=1,
            reconstructions_flat=reconstructions_flat,
            reconstructions_2d=reconstructions_2d,
            cpu_only=True,
            synthetic_verification_only=True,
        )
