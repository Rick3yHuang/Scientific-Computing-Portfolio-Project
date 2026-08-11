import pytest
import torch

from src.core import (
    compute_free_terms,
    create_orthonormal_frame,
    fit_conic,
    intersect_LOS_with_plane,
    master_function,
    normalize_normal_vector,
)


######################################################
# Fixtures for test data #############################
######################################################
@pytest.fixture
def fabricated_data() -> dict[str, torch.Tensor]:
    # random seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # generate a random rotation matrix using QR decomposition
    random_matrix = torch.randn(
        (3, 3),
        dtype=torch.float64,
        generator=generator,
    )
    rotation, _ = torch.linalg.qr(random_matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1

    # generate points on an ellipse in the XY-plane
    # one of the foci is at the origin
    angles = torch.linspace(0, 2 * torch.pi, 6, dtype=torch.float64)[:-1]
    semi_major_axis = torch.tensor(3.0, dtype=torch.float64)
    eccentricity = torch.sqrt(torch.tensor(5.0, dtype=torch.float64)) / semi_major_axis
    focal_distance = semi_major_axis * eccentricity
    semi_minor_axis = torch.sqrt(
        semi_major_axis.square() - focal_distance.square()
    )
    points_on_ellipse_star = torch.stack(
        (
            semi_major_axis * torch.cos(angles) - focal_distance,
            semi_minor_axis * torch.sin(angles),
            torch.zeros_like(angles),
        )
    )
    ellipse_coefficients = torch.stack(
        (
            semi_minor_axis.square(),
            torch.zeros((), dtype=torch.float64),
            semi_major_axis.square(),
            2 * semi_minor_axis.square() * focal_distance,
            torch.zeros((), dtype=torch.float64),
            semi_minor_axis.square() * focal_distance.square()
            - semi_major_axis.square() * semi_minor_axis.square(),
        )
    )
    conic_coefficients = ellipse_coefficients[[0, 2, 1, 3, 4]] / ellipse_coefficients[5]

    positions_star = torch.tensor(
        [
            [2.0, -1.0, 4.0, 3.0, -2.0],
            [1.0, 5.0, -3.0, 2.0, 6.0],
            [-4.0, 2.0, 1.0, -5.0, 3.0],
        ],
        dtype=torch.float64,
    )
    directions_star = points_on_ellipse_star - positions_star

    unit_normal = rotation[:, 2]
    if unit_normal[2] < 0:
        unit_normal = -unit_normal
    frame_3d = torch.stack((unit_normal, rotation[:, 0], rotation[:, 1]), dim=1)

    return {
        "ellipse_coefficients": ellipse_coefficients,
        "conic_coefficients": conic_coefficients,
        "points_on_ellipse": rotation @ points_on_ellipse_star,
        "positions": rotation @ positions_star,
        "directions": rotation @ directions_star,
        "frame_3d": frame_3d,
        "normal": unit_normal,
        "unit_normal": unit_normal,
    }


def master_function_inputs(
    fabricated_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        fabricated_data["normal"],
        fabricated_data["positions"],
        fabricated_data["directions"],
    )

######################################################
# Tests for normalize_normal_vector ##################
######################################################
def test_normalize_normal_vector_returns_unit_vector(fabricated_data) -> None:
    result = normalize_normal_vector(fabricated_data["normal"])
    torch.testing.assert_close(
        torch.linalg.vector_norm(result), torch.tensor(1.0, dtype=torch.float64)
    )


# Tests for exceptions
@pytest.mark.parametrize(
    ("invalid_normal", "message"),
    [
        (torch.zeros(3), "zero-length"),
        (torch.tensor([1.0, float("nan"), 2.0]), "non-finite"),
    ],
)


def test_normalize_normal_vector_rejects_invalid_input(
    invalid_normal: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_normal_vector(invalid_normal)


######################################################
# Tests for create_orthonormal_frame #################
######################################################
def test_create_orthonormal_frame_is_orthonormal(
    fabricated_data
) -> None:
    frame_3d = create_orthonormal_frame(fabricated_data["unit_normal"], torch.eye(3, dtype=fabricated_data["unit_normal"].dtype))
    dot_products = frame_3d.transpose(-2, -1) @ frame_3d
    identity = torch.eye(3, dtype=frame_3d.dtype)

    assert frame_3d.shape == (3, 3)
    torch.testing.assert_close(dot_products, identity)


######################################################
# Tests for intersect_LOS_with_plane #################
######################################################
def test_intersect_los_with_plane_returns_points_on_plane(
    fabricated_data
) -> None:
    positions = fabricated_data["positions"]
    directions = fabricated_data["directions"]
    frame_3d = fabricated_data["frame_3d"]

    intersections_2d = intersect_LOS_with_plane(
        directions, positions, frame_3d
    )
    intersections_3d = frame_3d[:, 1:] @ intersections_2d

    assert intersections_2d.shape == (2, 5)
    torch.testing.assert_close(
        fabricated_data["unit_normal"] @ intersections_3d, torch.zeros(5, dtype=torch.float64)
    )


# Tests for exceptions
def test_intersect_los_with_plane_rejects_parallel_line(fabricated_data) -> None:
    frame = torch.eye(3)
    directions = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    positions = torch.ones((3, 2))

    with pytest.raises(ValueError, match="parallel to the plane"):
        intersect_LOS_with_plane(directions, positions, frame)


######################################################
# Tests for fit_conic ################################
######################################################
def test_fit_conic_points_satisfy_conic_equation(fabricated_data) -> None:
    points = intersect_LOS_with_plane(
        fabricated_data["directions"],
        fabricated_data["positions"],
        fabricated_data["frame_3d"],
    )

    a, b, c, d, e = fit_conic(points)
    conic_coefficients = fabricated_data["conic_coefficients"]
    torch.testing.assert_close(
        torch.stack((a, b, c, d, e)), conic_coefficients, rtol=1e-10, atol=1e-10
    )

    x, y = points
    residuals = (
        a * x.square() + b * y.square() + c * x * y + d * x + e * y + 1
    )
    torch.testing.assert_close(
        residuals, torch.zeros_like(residuals), rtol=1e-10, atol=1e-10
    )


# Tests for exceptions
def test_fit_conic_rejects_wrong_number_of_points() -> None:
    with pytest.raises(ValueError, match=r"shape \(2, 5\)"):
        fit_conic(torch.zeros((2, 4)))


######################################################
# Tests for compute_free_terms #######################
######################################################
def test_compute_free_terms_matches_macaulay2_formula() -> None:
    coefficients = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0])

    result = compute_free_terms(coefficients)

    torch.testing.assert_close(result, torch.tensor([68.0, 67.0]))


def test_compute_free_terms_vanish_free_terms(fabricated_data) -> None:
    coefficients = fabricated_data["conic_coefficients"]

    result = compute_free_terms(coefficients)

    torch.testing.assert_close(result, torch.zeros(2, dtype=torch.float64))


def test_compute_free_terms_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(5,\)"):
        compute_free_terms(torch.zeros(4))


######################################################
# Tests for master_function ##########################
######################################################
def test_master_function_returns_zero_for_sampled_ellipse(
    fabricated_data
) -> None:
    normal_vector = fabricated_data["unit_normal"]
    position_matrix = fabricated_data["positions"]
    direction_matrix = fabricated_data["directions"]

    result = master_function(normal_vector, position_matrix, direction_matrix)

    torch.testing.assert_close(
        result, torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=1e-10
    )


# output tests
def test_master_function_returns_finite_float64_pair(fabricated_data) -> None:
    result = master_function(*master_function_inputs(fabricated_data))

    assert result.shape == (2,)
    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()


# Tests for exceptions
@pytest.mark.parametrize(
    ("input_index", "message"),
    [
        (1, r"pos_mat must have shape \(3, 5\)"),
        (2, r"dir_mat must have shape \(3, 5\)"),
    ],
)


def test_master_function_rejects_wrong_matrix_shape(
    fabricated_data, input_index: int, message: str
) -> None:
    inputs = list(master_function_inputs(fabricated_data))
    inputs[input_index] = torch.zeros((2, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match=message):
        master_function(*inputs)


@pytest.mark.parametrize("input_index", [0, 1, 2])


def test_master_function_rejects_mismatched_dtype(
    fabricated_data, input_index: int
    ) -> None:
    inputs = list(master_function_inputs(fabricated_data))
    inputs[input_index] = inputs[input_index].to(torch.float32)

    with pytest.raises(ValueError, match="same dtype"):
        master_function(*inputs)


def test_master_function_rejects_line_parallel_to_plane() -> None:
    normal_vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    position_matrix = torch.ones((3, 5), dtype=torch.float64)
    direction_matrix = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="parallel to the plane"):
        master_function(normal_vector, position_matrix, direction_matrix)
