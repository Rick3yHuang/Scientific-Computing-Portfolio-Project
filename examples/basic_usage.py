"""Minimal executable example for the Master Function pipeline."""

import torch

from src import master_function


def main() -> None:
    # Set up the input data for the master_function
    dtype = torch.float64
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(42)

    # fabricate a normal vector, position matrix, and direction matrix
    normal_vector = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    angles = torch.linspace(0, 2 * torch.pi, 6, dtype=dtype, device=device)[:-1]
    focal_distance = torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device))
    target_points = torch.stack(
        (
            3.0 * torch.cos(angles) - focal_distance,
            2.0 * torch.sin(angles),
            torch.zeros_like(angles),
        )
    )
    position_matrix = torch.randn(
        (3, 5),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    rotation, _ = torch.linalg.qr(
        torch.randn((3, 3), dtype=dtype, device=device, generator=generator)
    )
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1

    normal_vector = rotation @ normal_vector
    target_points = rotation @ target_points
    position_matrix = rotation @ position_matrix
    direction_matrix = target_points - position_matrix

    # Call the master_function with the fabricated data
    free_terms = master_function(
        normal_vector, position_matrix, direction_matrix
    )

    # Print the results
    for name, tensor in (
        ("normal_vector", normal_vector),
        ("position_matrix", position_matrix),
        ("direction_matrix", direction_matrix),
    ):
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"output (free terms): {free_terms}")
    print(f"output shape: {tuple(free_terms.shape)}")
    print(f"output norm: {torch.linalg.vector_norm(free_terms)}")


if __name__ == "__main__":
    main()
