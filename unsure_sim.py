from typing import Tuple

import numpy as np
from numba import jit
from SALib.sample import sobol_sequence
from scipy.integrate import solve_ivp


def normalize_dataset(
    m: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bounds: Tuple[float, float]
) -> np.ndarray:
    x = maxs - mins
    y = bounds[1] - bounds[0]
    m -= mins - bounds[0] * x / y
    m *= y / x
    return m


def denormalize_dataset(
    m: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bounds: Tuple[float, float]
) -> np.ndarray:
    x = bounds[1] - bounds[0]
    y = maxs - mins
    m -= bounds[0] - mins * x / y
    m *= y / x
    return m


@jit(nopython=True, error_model="numpy")
def state_derivative(
    t: float,
    y: np.ndarray,
    A: int,
    B: int,
    D: int,
    sample_coefs: np.ndarray,
    sample_coefs_sum: float,
    bounds: Tuple[float, float],
    empty_like_state: np.ndarray,
    EPS: float = 1e-7,
    I2P_COEF: float = 1e0,
    WALL_COEF: float = 1e-1,
    FRIC_COEF: float = 1.0,
) -> np.ndarray:
    state = y.reshape(A + B, 2, D)
    prev_pos: np.ndarray = state[:, 0, :]

    # all_pos: (1, A+B, D)
    all_pos = np.expand_dims(prev_pos, axis=0)
    # unsure_pos: (B, 1, D)
    unsure_pos = np.expand_dims(prev_pos[A:], axis=1)
    # all_to_unsure: (B, A+B, D)
    all_to_unsure = unsure_pos - all_pos
    # dist: (B, A+B)
    dist: np.ndarray = np.sqrt(np.sum(all_to_unsure ** 2, axis=2)) + EPS
    all_to_unsure /= np.expand_dims(dist, axis=2)
    # force_mag: (B, A+B)
    force_mag = (1 / dist ** 2) * sample_coefs * I2P_COEF
    all_to_unsure *= np.expand_dims(force_mag, axis=2)
    all_to_unsure /= sample_coefs_sum
    for i in range(B):
        all_to_unsure[i, A + i] = 0
    # all_to_unsure: (B, D)
    all_to_unsure = np.sum(all_to_unsure, axis=1)

    unsure_pos = prev_pos[A:]
    # dist: (B, D)
    dist = unsure_pos - bounds[0]
    wall_to_unsure = np.ones_like(dist)
    up_force_mag = np.exp(-dist / WALL_COEF)
    dist = bounds[1] - unsure_pos
    down_force_mag = np.exp(-dist / WALL_COEF)
    wall_to_unsure *= up_force_mag - down_force_mag

    unsure_vel = state[A:, 1, :]
    friction = -FRIC_COEF * unsure_vel

    delta: np.ndarray = empty_like_state
    delta[:, 0, :] = state[:, 1, :]
    delta[:A, 1, :] = 0
    delta[A:, 1, :] = all_to_unsure + wall_to_unsure + friction
    return delta.reshape(-1)


class UnsureSimulator:
    def __init__(
        self,
        dataset: np.ndarray,
        cls_coefs: np.ndarray,
        unsure_ratio: float,
        bounds: Tuple[float, float] = (-10.0, 10.0),
    ):
        sure_particles: np.ndarray = dataset[:, :-1].copy()
        self.a, self.d = sure_particles.shape
        self.mins, self.maxs = (
            np.expand_dims(np.min(sure_particles, axis=0), axis=0),
            np.expand_dims(np.max(sure_particles, axis=0), axis=0),
        )
        self.bounds = bounds
        sure_particles = normalize_dataset(sure_particles, self.mins, self.maxs, bounds)
        sure_vel: np.ndarray = np.zeros_like(sure_particles)
        sure_states: np.ndarray = np.stack([sure_particles, sure_vel], axis=1)
        del sure_particles, sure_vel

        unsure_particles: np.ndarray = normalize_dataset(
            sobol_sequence.sample(int(self.a * unsure_ratio), self.d),
            np.array(0.0),
            np.array(1.0),
            (
                bounds[0] + (bounds[1] - bounds[0]) / 1000,
                bounds[1] - (bounds[1] - bounds[0]) / 1000,
            ),
        )
        self.b = unsure_particles.shape[0]
        vel_mean = (bounds[1] - bounds[0]) / 4
        vel_std = (bounds[1] - bounds[0]) / 16
        unsure_vel: np.ndarray = (
            np.random.randn(*unsure_particles.shape) * vel_std + vel_mean
        ) * np.random.choice([-1, 1], size=(*unsure_particles.shape,))
        unsure_state: np.ndarray = np.stack(
            [
                unsure_particles.reshape(unsure_particles.shape[0], self.d),
                unsure_vel.reshape(unsure_particles.shape[0], self.d),
            ],
            axis=1,
        )
        del unsure_particles, unsure_vel

        self.state: np.ndarray = np.concatenate((sure_states, unsure_state), axis=0)
        self.sample_coefs: np.ndarray = np.empty((1, self.a + self.b))
        self.sample_coefs[0, : self.a] = 1 / cls_coefs[dataset[:, -1].astype(int)]
        self.sample_coefs[0, self.a :] = 1
        self.sample_coefs_sum = np.sum(self.sample_coefs)
        self.b_diag_indices = np.diag_indices(self.b, ndim=2)
        self.empty_like_state = np.empty_like(self.state)

    def simulate(self) -> np.ndarray:
        sol = solve_ivp(
            state_derivative,
            [0, 10],
            self.state.reshape(-1),
            method="RK23",
            args=(
                self.a,
                self.b,
                self.d,
                self.sample_coefs,
                self.sample_coefs_sum,
                self.bounds,
                self.empty_like_state,
            ),
        )
        last_state: np.ndarray = sol.y[:, -1].reshape(self.a + self.b, 2, self.d)
        return denormalize_dataset(
            last_state[self.a :, 0, :].reshape(self.b, self.d),
            self.mins,
            self.maxs,
            self.bounds,
        )
