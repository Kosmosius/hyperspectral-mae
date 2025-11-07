"""Simple physics renderer implemented without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from hsi_fm.physics.srf import SpectralResponseFunction, convolve_srf, gaussian_srf


Number = float


VectorLike = Sequence[Number]
MatrixLike = Sequence[VectorLike]


def _ensure_2d(array: Union[VectorLike, MatrixLike]) -> List[List[Number]]:
    array_list = list(array)  # type: ignore[arg-type]
    if not array_list:
        return []
    first = array_list[0]
    if isinstance(first, (list, tuple)):
        return [list(row) for row in array_list]  # type: ignore[list-item]
    return [list(array_list)]  # type: ignore[list-item]


def _matmul(a: List[List[Number]], b: List[List[Number]]) -> List[List[Number]]:
    if not a or not b:
        return []
    result: List[List[Number]] = []
    b_t = list(map(list, zip(*b)))
    for row in a:
        result.append([sum(x * y for x, y in zip(row, col)) for col in b_t])
    return result


def _transpose(matrix: List[List[Number]]) -> List[List[Number]]:
    return list(map(list, zip(*matrix))) if matrix else []


def _identity(n: int) -> List[List[Number]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _augment(a: List[List[Number]], b: List[List[Number]]) -> List[List[Number]]:
    return [row_a + row_b for row_a, row_b in zip(a, b)]


def _gauss_jordan(matrix: List[List[Number]]) -> List[List[Number]]:
    n = len(matrix)
    m = len(matrix[0])
    for col in range(n):
        pivot = matrix[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")
        inv_pivot = 1.0 / pivot
        matrix[col] = [val * inv_pivot for val in matrix[col]]
        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [val - factor * pivot_val for val, pivot_val in zip(matrix[row], matrix[col])]
    return [row[n:] for row in matrix]


def _invert(matrix: List[List[Number]]) -> List[List[Number]]:
    n = len(matrix)
    identity = _identity(n)
    augmented = _augment([row[:] for row in matrix], identity)
    return _gauss_jordan(augmented)


@dataclass
class AtmosphericState:
    irradiance: Number = 1.0
    transmittance: Number = 1.0
    path_radiance: Number = 0.0


class LabToSensorRenderer:
    def __init__(self, srf: SpectralResponseFunction):
        self.srf = srf

    def render(
        self,
        reflectance: Sequence[Sequence[Number] | Number],
        wavelengths: Sequence[Number],
        atmosphere: Optional[AtmosphericState] = None,
    ) -> List[List[Number]]:
        atmosphere = atmosphere or AtmosphericState()
        refl = _ensure_2d(reflectance)
        if refl and len(refl[0]) != len(wavelengths):
            raise ValueError("Reflectance grid mismatch")
        radiance_hr = [
            [atmosphere.irradiance * val * atmosphere.transmittance + atmosphere.path_radiance for val in row]
            for row in refl
        ]
        return convolve_srf(radiance_hr, wavelengths, self.srf)

    def invert(
        self,
        radiance: Sequence[Sequence[Number] | Number],
        wavelengths: Sequence[Number],
        atmosphere: Optional[AtmosphericState] = None,
    ) -> List[List[Number]]:
        atmosphere = atmosphere or AtmosphericState()
        rad = _ensure_2d(radiance)
        if not rad:
            return []
        if self.srf.responses is None:
            responses = gaussian_srf(self.srf.centers, self.srf.widths, wavelengths)
        else:
            responses = [row[:] for row in self.srf.responses]
        responses_t = _transpose(responses)
        gram = _matmul(responses, responses_t)
        epsilon = 1e-6
        for i in range(len(gram)):
            gram[i][i] += epsilon
        gram_inv = _invert(gram)
        pseudo = _matmul(responses_t, gram_inv)
        scale = atmosphere.irradiance * atmosphere.transmittance + 1e-12
        reflectance: List[List[Number]] = []
        for measurement in rad:
            y_col = [[val - atmosphere.path_radiance] for val in measurement]
            x_col = _matmul(pseudo, y_col)
            row = [val[0] / scale for val in x_col]
            reflectance.append(row)
        return reflectance


__all__ = ["LabToSensorRenderer", "AtmosphericState"]
