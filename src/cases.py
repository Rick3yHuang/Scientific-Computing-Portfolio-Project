"""Reusable deterministic cases for Master Function evaluation and solvers."""

from dataclasses import dataclass
from typing import Literal

import torch


CaseStatus = Literal["root", "degenerate"]


@dataclass(frozen=True)
class MasterFunctionCase:
	"""Fixed inputs and acceptance criteria for one Master Function case."""

	case_id: str
	normal_vector: torch.Tensor
	position_matrix: torch.Tensor
	direction_matrix: torch.Tensor
	dtype: torch.dtype
	expected_status: CaseStatus
	reference_root: torch.Tensor | None
	residual_tolerance: float


def _build_fabricated_ellipse_root_case() -> MasterFunctionCase:
	dtype = torch.float64
	generator = torch.Generator().manual_seed(42)

	random_matrix = torch.randn((3, 3), dtype=dtype, generator=generator)
	rotation, _ = torch.linalg.qr(random_matrix)
	if torch.linalg.det(rotation) < 0:
		rotation[:, -1] *= -1

	angles = torch.linspace(0, 2 * torch.pi, 6, dtype=dtype)[:-1]
	focal_distance = torch.sqrt(torch.tensor(5.0, dtype=dtype))
	target_points = torch.stack(
		(
			3.0 * torch.cos(angles) - focal_distance,
			2.0 * torch.sin(angles),
			torch.zeros_like(angles),
		)
	)
	position_matrix = torch.tensor(
		[
			[2.0, -1.0, 4.0, 3.0, -2.0],
			[1.0, 5.0, -3.0, 2.0, 6.0],
			[-4.0, 2.0, 1.0, -5.0, 3.0],
		],
		dtype=dtype,
	)

	normal_vector = rotation[:, 2]
	if normal_vector[2] < 0:
		normal_vector = -normal_vector

	position_matrix = rotation @ position_matrix
	direction_matrix = rotation @ target_points - position_matrix

	return MasterFunctionCase(
		case_id="fabricated_ellipse_root",
		normal_vector=normal_vector,
		position_matrix=position_matrix,
		direction_matrix=direction_matrix,
		dtype=dtype,
		expected_status="root",
		reference_root=normal_vector,
		residual_tolerance=1e-10,
	)


def _build_parallel_line_of_sight_case() -> MasterFunctionCase:
	dtype = torch.float64
	return MasterFunctionCase(
		case_id="parallel_line_of_sight",
		normal_vector=torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
		position_matrix=torch.ones((3, 5), dtype=dtype),
		direction_matrix=torch.tensor(
			[
				[1.0, 1.0, 0.0, -1.0, 1.0],
				[0.0, 0.0, 1.0, 0.0, -1.0],
				[-1.0, 0.0, -1.0, -1.0, -1.0],
			],
			dtype=dtype,
		),
		dtype=dtype,
		expected_status="degenerate",
		reference_root=None,
		residual_tolerance=0.0,
	)


_CASE_REGISTRY = {
	"fabricated_ellipse_root": _build_fabricated_ellipse_root_case(),
	"parallel_line_of_sight": _build_parallel_line_of_sight_case(),
}


def get_master_function_case(case_id: str) -> MasterFunctionCase:
	"""Return an independent copy of a registered deterministic case."""
	case = _CASE_REGISTRY[case_id]
	return MasterFunctionCase(
		case_id=case.case_id,
		normal_vector=case.normal_vector.clone(),
		position_matrix=case.position_matrix.clone(),
		direction_matrix=case.direction_matrix.clone(),
		dtype=case.dtype,
		expected_status=case.expected_status,
		reference_root=(
			None if case.reference_root is None else case.reference_root.clone()
		),
		residual_tolerance=case.residual_tolerance,
	)
