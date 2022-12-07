"""
Iterative closest point algorithm accelerated by CUDA.
"""

from argparse import ArgumentParser
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.neighbors import NearestNeighbors
from tqdm import tqdm


class ICP(object):
    """Iterative closest point algorithm."""

    def __init__(
        self,
        source,
        target,
        percentile=0.5,
        max_steps=300,
        early_stop_eps=1e-4,
        early_stop_steps=10,
    ):
        super().__init__()
        self.source = cp.array(source)
        self.target = cp.array(target)
        self.percentile = percentile
        self.max_steps = max_steps
        self.early_stop_eps = early_stop_eps
        self.early_stop_steps = early_stop_steps
        self.rotation = cp.eye(source.shape[1])
        self.translation = cp.zeros(source.shape[1])
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(target)

    @staticmethod
    def transform(points, rotation, translation):
        """Transform points according to rotation and translation."""
        return points @ rotation.T + translation.T

    def find_correrspondences(self, rotation, translation):
        """Find nearest neighbors from transformed source to target."""
        transformed_source = self.transform(self.source, rotation, translation)
        distances, indices = self.nearest_neighbors.kneighbors(
            transformed_source, return_distance=True
        )
        distances = distances.flatten()
        indices = indices.flatten()
        return distances, indices

    def topk_correrspondences(self, distances, indices):
        """Select the topk correrspondences with minimum distances."""
        topk = int(self.percentile * len(distances))
        partition = cp.argpartition(distances, topk)
        selected = partition[:topk]
        selected_distances = distances[selected]
        selected_indices = indices[selected]
        selected_source = self.source[selected]
        selected_target = self.target[selected_indices]
        return selected_distances, selected_indices, selected_source, selected_target

    @staticmethod
    def find_transformation(source, target):
        """Find the best transformation from source to target."""
        source_mean = cp.mean(source, axis=0, keepdims=True)
        target_mean = cp.mean(target, axis=0, keepdims=True)
        source_diff = source - source_mean
        target_diff = target - target_mean
        H = source_diff.T @ target_diff
        U, S, VT = cp.linalg.svd(H)
        sign = cp.sign(cp.linalg.det(VT.T @ U.T))
        diag = cp.eye(source.shape[1])
        diag[-1, -1] = sign
        rotation = (VT.T @ diag) @ U.T
        translation = target_mean.T - rotation @ source_mean.T
        translation = translation.flatten()
        return rotation, translation

    def step(self):
        """Perform a single step of NaiveICP algorithm."""
        distances, indices = self.find_correrspondences(self.rotation, self.translation)
        (
            selected_distances,
            selected_indices,
            selected_source,
            selected_target,
        ) = self.topk_correrspondences(distances, indices)
        self.rotation, self.translation = self.find_transformation(
            selected_source, selected_target
        )
        topk_distance = cp.max(selected_distances)
        mean_distance = cp.mean(selected_distances)
        return topk_distance, mean_distance

    def run(self):
        """Run NaiveICP algorithm for multiple steps."""
        early_stop_counter = 0
        best_topk_distance = cp.inf
        with tqdm(total=self.max_steps) as pbar:
            for step in range(self.max_steps):
                topk_distance, mean_distance = self.step()
                pbar.set_description_str(
                    f"topk_distance: {topk_distance:.4f}, mean_distance: {mean_distance:.4f}"
                )
                pbar.update(1)
                # early stopping
                if topk_distance >= best_topk_distance - self.early_stop_eps:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                if early_stop_counter >= self.early_stop_steps:
                    pbar.close()
                    print("Early stopping")
                    break
                best_topk_distance = min(topk_distance, best_topk_distance)
        transformed_source = self.transform(
            self.source, self.rotation, self.translation
        )
        return transformed_source.get(), self.rotation.get(), self.translation.get()


if __name__ == "__main__":
    parser = ArgumentParser("Naive iterative closest point algorithm.")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--early-stop-eps", type=float, default=1e-4)
    parser.add_argument("--early-stop-steps", type=int, default=10)
    args = parser.parse_args()

    source = np.loadtxt(args.source, comments="#")
    target = np.loadtxt(args.target, comments="#")

    icp = ICP(
        source,
        target,
        percentile=args.percentile,
        max_steps=args.max_steps,
        early_stop_eps=args.early_stop_eps,
        early_stop_steps=args.early_stop_steps,
    )
    transformed_source, rotation, translation = icp.run()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, transformed_source)
    print(f"Transformed points saved to {args.output}")
