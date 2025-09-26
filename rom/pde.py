"""Equation abstractions for Galerkin ROM construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class ModeDerivatives:
    """Container for POD modes and their spatial derivatives."""

    phi_p: ArrayLike
    phi_u: ArrayLike
    phi_v: ArrayLike
    dx_phi_p: ArrayLike
    dy_phi_p: ArrayLike
    dx_phi_u: ArrayLike
    dy_phi_u: ArrayLike
    dx_phi_v: ArrayLike
    dy_phi_v: ArrayLike
    lap_phi_u: ArrayLike
    lap_phi_v: ArrayLike

    @property
    def num_modes(self) -> int:
        return self.phi_u.shape[1]


@dataclass
class BoundaryData:
    """Boundary condition data and derivatives."""

    u_bc: ArrayLike
    dx_u_bc: ArrayLike
    dy_u_bc: ArrayLike
    lap_u_bc: ArrayLike


@dataclass
class PDEContext:
    """Aggregates data required to assemble reduced operators."""

    modes: ModeDerivatives
    boundary: BoundaryData
    inner_product: Callable[[ArrayLike, ArrayLike], float]

    def dot(self, lhs: ArrayLike, rhs: ArrayLike) -> float:
        """Evaluate the configured inner product between two fields."""
        return self.inner_product(lhs, rhs)


class PDEDefinition:
    """Interface for computing ROM tensors of a PDE."""

    name: str = "abstract-pde"

    def compute_constant_terms(self, ctx: PDEContext) -> Tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError

    def compute_linear_terms(self, ctx: PDEContext) -> Tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError

    def compute_quadratic_terms(self, ctx: PDEContext) -> ArrayLike:
        raise NotImplementedError


class SteadyNavierStokesPDE(PDEDefinition):
    """Steady incompressible Navier–Stokes in primitive variables."""

    name = "steady-incompressible-navier-stokes"

    def compute_constant_terms(self, ctx: PDEContext) -> Tuple[ArrayLike, ArrayLike]:
        K = ctx.modes.num_modes
        C1 = np.zeros(K)
        C2 = np.zeros(K)

        for m in range(K):
            phi_p_m = ctx.modes.phi_p[:, m]
            phi_u_m = ctx.modes.phi_u[:, m]

            C1[m] = ctx.dot(ctx.boundary.dx_u_bc, phi_p_m) + ctx.dot(
                ctx.boundary.u_bc * ctx.boundary.dx_u_bc, phi_u_m
            )
            C2[m] = ctx.dot(-ctx.boundary.lap_u_bc, phi_u_m)

        return C1, C2

    def compute_linear_terms(self, ctx: PDEContext) -> Tuple[ArrayLike, ArrayLike]:
        K = ctx.modes.num_modes
        L1 = np.zeros((K, K))
        L2 = np.zeros((K, K))

        for m in range(K):
            phi_p_m = ctx.modes.phi_p[:, m]
            phi_u_m = ctx.modes.phi_u[:, m]
            phi_v_m = ctx.modes.phi_v[:, m]

            for j in range(K):
                L1_rc = ctx.dot(
                    ctx.modes.dx_phi_u[:, j] + ctx.modes.dy_phi_v[:, j],
                    phi_p_m,
                )
                L1_ru = ctx.dot(
                    ctx.boundary.u_bc * ctx.modes.dx_phi_u[:, j]
                    + ctx.modes.phi_u[:, j] * ctx.boundary.dx_u_bc
                    + ctx.modes.phi_v[:, j] * ctx.boundary.dy_u_bc
                    + ctx.modes.dx_phi_p[:, j],
                    phi_u_m,
                )
                L1_rv = ctx.dot(
                    ctx.boundary.u_bc * ctx.modes.dx_phi_v[:, j]
                    + ctx.modes.dy_phi_p[:, j],
                    phi_v_m,
                )
                L1[m, j] = L1_rc + L1_ru + L1_rv

                L2_ru = ctx.dot(-ctx.modes.lap_phi_u[:, j], phi_u_m)
                L2_rv = ctx.dot(-ctx.modes.lap_phi_v[:, j], phi_v_m)
                L2[m, j] = L2_ru + L2_rv

        return L1, L2

    def compute_quadratic_terms(self, ctx: PDEContext) -> ArrayLike:
        K = ctx.modes.num_modes
        Q = np.zeros((K, K, K))

        for m in range(K):
            phi_u_m = ctx.modes.phi_u[:, m]
            phi_v_m = ctx.modes.phi_v[:, m]

            for j in range(K):
                dx_u_j = ctx.modes.dx_phi_u[:, j]
                dy_u_j = ctx.modes.dy_phi_u[:, j]
                dx_v_j = ctx.modes.dx_phi_v[:, j]
                dy_v_j = ctx.modes.dy_phi_v[:, j]

                for i in range(K):
                    quad_u = ctx.modes.phi_u[:, i] * dx_u_j + ctx.modes.phi_v[:, i] * dy_u_j
                    quad_v = ctx.modes.phi_u[:, i] * dx_v_j + ctx.modes.phi_v[:, i] * dy_v_j

                    Q[m, i, j] = ctx.dot(quad_u, phi_u_m) + ctx.dot(quad_v, phi_v_m)

        return Q
