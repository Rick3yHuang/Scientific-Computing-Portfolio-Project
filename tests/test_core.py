import unittest

import torch

from src.core import (
    compute_free_terms,
    create_orthonormal_frame,
    fit_conic,
    intersect_LOS_with_plane,
    master_function,
    normalize_normal_vector,
)

class TestNormalizeNormalVector(unittest.TestCase):
    def test_returns_unit_vector(self) -> None:
        normal = torch.tensor([3.0, 4.0, 0.0])

        result = normalize_normal_vector(normal)

        expected = torch.tensor([0.6, 0.8, 0.0])
        torch.testing.assert_close(result, expected)

    def test_result_has_length_one(self) -> None:
        normal = torch.tensor([2.0, 3.0, 6.0])

        result = normalize_normal_vector(normal)

        length = torch.linalg.vector_norm(result)
        torch.testing.assert_close(length, torch.tensor(1.0))

    def test_zero_vector_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length"):
            normalize_normal_vector(torch.zeros(3))

class TestCreateOrthonormalFrame(unittest.TestCase):
    def test_orthonormality(self) -> None:
        normal = torch.tensor([0.0, 0.0, 1.0])
        dir_mat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        frame = create_orthonormal_frame(normal, dir_mat)

        self.assertEqual(frame.shape, (3, 3))
        dot_products = torch.matmul(frame.transpose(-2, -1), frame)
        identity = torch.eye(3, dtype=frame.dtype)
        torch.testing.assert_close(dot_products, identity)


class TestIntersectLOSWithPlane(unittest.TestCase):
    def test_intersects_each_column_with_origin_centered_plane(self) -> None:
        positions = torch.tensor(
            [[1.0, -2.0, 3.0, 4.0, -5.0], 
             [6.0, 7.0, -8.0, 9.0, 10.0], 
             [11.0, -12.0, 13.0, 14.0, -15.0]]
        )
        directions = torch.tensor(
            [[2.0, -1.0, 4.0, 3.0, -2.0], 
             [1.0, 5.0, -3.0, 2.0, 6.0], 
             [-4.0, 2.0, 1.0, -5.0, 3.0]]
        )
        normal = normalize_normal_vector(torch.tensor([1.0, -2.0, 3.0]))
        
        frame_3d = create_orthonormal_frame(normal, directions)

        intersections_2D = intersect_LOS_with_plane(directions, positions, frame_3d)
        intersections_3D = frame_3d[:, 1:] @ intersections_2D

        self.assertEqual(intersections_2D.shape, (2, 5))
        torch.testing.assert_close(normal @ intersections_3D, torch.zeros(5))


class TestFitConic(unittest.TestCase):
    def test_fitted_points_satisfy_conic_equation(self) -> None:
        points = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0, 0.6],
                [0.0, 0.0, 1.0, -1.0, 0.8],
            ],
            dtype=torch.float64,
        )
        frame_3d = torch.eye(3, dtype=torch.float64)

        coefficients = fit_conic(points, frame_3d)
        a, b, c, d, e = coefficients
        x, y = points

        residuals = (
            a * x.square()
            + b * y.square()
            + c * x * y
            + d * x
            + e * y
            + 1
        )

        torch.testing.assert_close(
            residuals,
            torch.zeros_like(residuals),
            rtol=1e-10,
            atol=1e-10,
        )


class TestComputeFreeTerms(unittest.TestCase):
    def test_matches_the_macaulay2_formula(self) -> None:
        coefficients = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0])

        free_terms = compute_free_terms(coefficients)

        torch.testing.assert_close(free_terms, torch.tensor([68.0, 67.0]))


class TestMasterFunction(unittest.TestCase):
    def test_returns_free_terms_for_a_sampled_ellipse(self) -> None:
        normal = normalize_normal_vector(
            torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
        )
        directions = torch.tensor(
            [
                [1.0, -2.0, 3.0, 4.0, -1.0],
                [2.0, 1.0, -3.0, 2.0, 5.0],
                [-1.0, -2.0, -1.0, -3.0, -2.0],
            ],
            dtype=torch.float64,
        )
        frame_3d = create_orthonormal_frame(normal, directions)
        angles = torch.tensor(
            [0.0, 0.7, 1.8, 3.4, 5.1], dtype=torch.float64
        )
        semi_major_axis = torch.tensor(0.5, dtype=torch.float64)
        semi_minor_axis = torch.tensor(1.0 / 3.0, dtype=torch.float64)
        focal_distance = torch.sqrt(semi_major_axis.square() - semi_minor_axis.square())
        ellipse_center = torch.stack((focal_distance, torch.zeros_like(focal_distance)))
        ellipse_coordinates = torch.stack(
            (
                semi_major_axis * torch.cos(angles),
                semi_minor_axis * torch.sin(angles),
            )
        ) + ellipse_center[:, None]
        intersection_points = frame_3d[:, 1:] @ ellipse_coordinates
        positions = intersection_points - directions

        free_terms = master_function(normal, positions, directions)

        torch.testing.assert_close(
            free_terms, torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=1e-10
        )


if __name__ == "__main__":
	unittest.main()
