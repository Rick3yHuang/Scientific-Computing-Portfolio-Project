import pytest
import torch

from src.cases import get_master_function_case
from src.core import master_function


def test_fabricated_ellipse_root_case_matches_its_contract() -> None:
    case = get_master_function_case("fabricated_ellipse_root")

    assert case.expected_status == "root"
    assert case.normal_vector.shape == (3,)
    assert case.position_matrix.shape == (3, 5)
    assert case.direction_matrix.shape == (3, 5)
    assert case.reference_root is not None
    assert case.reference_root.shape == (3,)

    for tensor in (
        case.normal_vector,
        case.position_matrix,
        case.direction_matrix,
        case.reference_root,
    ):
        assert tensor.dtype == case.dtype
        assert tensor.device.type == "cpu"

    result = master_function(
        case.normal_vector, case.position_matrix, case.direction_matrix
    )
    assert result.shape == (2,)
    assert torch.isfinite(result).all()
    assert torch.linalg.vector_norm(result) <= case.residual_tolerance


def test_parallel_line_of_sight_case_matches_its_contract() -> None:
    case = get_master_function_case("parallel_line_of_sight")

    assert case.expected_status == "degenerate"
    assert case.reference_root is None
    assert case.normal_vector.shape == (3,)
    assert case.position_matrix.shape == (3, 5)
    assert case.direction_matrix.shape == (3, 5)

    for tensor in (
        case.normal_vector,
        case.position_matrix,
        case.direction_matrix,
    ):
        assert tensor.dtype == case.dtype
        assert tensor.device.type == "cpu"

    with pytest.raises(ValueError, match="parallel to the plane"):
        master_function(
            case.normal_vector, case.position_matrix, case.direction_matrix
        )


def test_case_lookup_returns_independent_tensor_copies() -> None:
    first_case = get_master_function_case("fabricated_ellipse_root")
    second_case = get_master_function_case("fabricated_ellipse_root")

    first_case.position_matrix[0, 0] = 999.0

    assert second_case.position_matrix[0, 0] != 999.0


def test_case_lookup_rejects_unknown_case_id() -> None:
    with pytest.raises(KeyError, match="unknown_case"):
        get_master_function_case("unknown_case")