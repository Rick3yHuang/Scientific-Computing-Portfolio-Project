"""Minimal executable example for the Master Function pipeline."""

import torch

from src import master_function
from src.cases import get_master_function_case


def main() -> None:
    case = get_master_function_case("fabricated_ellipse_root")
    free_terms = master_function(
        case.normal_vector, case.position_matrix, case.direction_matrix
    )

    print(f"case id: {case.case_id}")
    for name, tensor in (
        ("normal_vector", case.normal_vector),
        ("position_matrix", case.position_matrix),
        ("direction_matrix", case.direction_matrix),
    ):
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"output (free terms): {free_terms}")
    print(f"output shape: {tuple(free_terms.shape)}")
    print(f"output norm: {torch.linalg.vector_norm(free_terms)}")
    print(f"expected status: {case.expected_status}")
    print(f"residual tolerance: {case.residual_tolerance}")


if __name__ == "__main__":
    main()
