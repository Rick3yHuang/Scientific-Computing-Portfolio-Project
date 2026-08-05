"""Minimal executable example for the Master Function pipeline."""

import torch

from src import master_function


def main() -> None:
    normal_vectors = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]],
        dtype=torch.float64,
    )
    normalized = master_function(normal_vectors)
    print(normalized)


if __name__ == "__main__":
    main()
