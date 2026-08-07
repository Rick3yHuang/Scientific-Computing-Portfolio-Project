"""Core numerical operations for the Master Function project."""

import torch

def normalize_normal_vector(n: torch.Tensor) -> torch.Tensor:
    """
    return a normalized version of the input normal vector(s).
    """
    magnitude = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    if not torch.all(torch.isfinite(magnitude)):
        raise ValueError("Input contains non-finite values.")
    if not torch.is_floating_point(n):
        raise TypeError("Input must be a floating-point or complex tensor.")
    if torch.any(magnitude == 0):
        raise ValueError("Input contains zero-length vectors.")
    return n / magnitude

def create_orthonormal_frame(normal: torch.Tensor, dir_mat: torch.Tensor) -> torch.Tensor:
    """
    Create an orthonormal frame from a unit normal vector and a direction matrix.

    ``normal`` is a 3D vector. ``dir_mat`` is a 3xN matrix of N 3D direction vectors.
    The returned tensor is a 3x3 orthonormal matrix whose first column is the normal vector, and the other two columns are orthonormal vectors in the plane defined by the normal vector and the first direction vector in ``dir_mat``.
    """
    assert torch.isclose(torch.linalg.vector_norm(normal), torch.tensor(1.0, dtype=normal.dtype, device=normal.device)), "Normal vector must be a unit vector."
    dir_0 = dir_mat[:, 0]
    v_1 = normalize_normal_vector(torch.cross(normal, dir_0,dim=-1))
    v_2 = normalize_normal_vector(torch.cross(normal, v_1, dim=-1))
    return torch.stack([normal, v_1, v_2], dim=-1)


def intersect_LOS_with_plane(dir_mat: torch.Tensor, pos_mat: torch.Tensor, frame_3D: torch.Tensor) -> torch.Tensor:
    """
    Intersect 3D lines of sight with the plane through the origin.

    ``dir_mat`` and ``pos_mat`` contain one 3D vector per column. The returned
    tensor has one intersection point per column.

    ``frame_3D`` is a 3x3 orthonormal matrix whose first column is the normal vector of the plane.
    """
    if dir_mat.shape != pos_mat.shape or dir_mat.ndim != 2 or dir_mat.shape[0] != 3:
        raise ValueError("dir_mat and pos_mat must both have shape (3, N).")
    if frame_3D.shape != (3, 3):
        raise ValueError("frame_3D must have shape (3, 3).")

    w = frame_3D[:, 0]
    numerators = -(w @ pos_mat)
    denominators = w @ dir_mat
    if torch.any(denominators == 0):
        raise ValueError("A line of sight is parallel to the plane.")
    intersections_3D = pos_mat + dir_mat * (numerators / denominators)
    intersections_2D = frame_3D[:, 1:].transpose(0, 1) @ intersections_3D
    return intersections_2D


def fit_conic(intersections_2D: torch.Tensor) -> torch.Tensor:
    """
    Fit ``ax^2 + by^2 + cxy + dx + ey + 1 = 0`` through five points.

    ``intersections_2D`` has shape ``(2, 5)`` with one 2D point per column.
    The returned tensor has shape ``(5,)`` in ``(x^2, y^2, xy, x, y)`` coefficient order.
    """
    if intersections_2D.shape != (2, 5):
        raise ValueError("intersections_2D must have shape (2, 5).")
    
    [x_coordinates, y_coordinates] = intersections_2D
    source_matrix = torch.stack(
        (
            x_coordinates.square(),
            y_coordinates.square(),
            x_coordinates * y_coordinates,
            x_coordinates,
            y_coordinates,
        ),
        dim=1,
    )
    return torch.linalg.solve(source_matrix, -torch.ones(5, dtype=intersections_2D.dtype, device=intersections_2D.device))

def compute_free_terms(conic_coefficients: torch.Tensor) -> torch.Tensor:
    """Compute the two free terms from conic coefficients ``(a, b, c, d, e)``."""
    if conic_coefficients.shape != (5,):
        raise ValueError("conic_coefficients must have shape (5,).")

    a, b, c, d, e = conic_coefficients
    return torch.stack(
        (
            e.square() - 4 * b - d.square() + 4 * a,
            d * e - 2 * c,
        )
    )

def master_function(normal_vectors: torch.Tensor, pos_mat: torch.Tensor, dir_mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the two free terms from the conic coefficients fitted to the intersection points of the lines of sight with the plane defined by the normal vectors.

    ``normal_vectors`` is a 3D vector. ``pos_mat`` and ``dir_mat`` are 3xN matrices of N 3D position and direction vectors, respectively.
    """
    w = normalize_normal_vector(normal_vectors)
    frame_3D = create_orthonormal_frame(w, dir_mat)
    intersections_2D = intersect_LOS_with_plane(dir_mat, pos_mat, frame_3D)
    conic_coefficients = fit_conic(intersections_2D)
    free_terms = compute_free_terms(conic_coefficients)
    return free_terms
