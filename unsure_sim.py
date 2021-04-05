import numpy as np
from SALib.sample import sobol_sequence
from scipy.integrate import solve_ivp


class UnsureSimulator:
    EPS = 1e-5
    I2P_COEF = 1e-1
    WALL_COEF = 1e-4
    FRIC_COEF = 0.5

    def __init__(self, dataset: np.ndarray, cls_coefs: np.ndarray, unsure_ratio: float):
        sure_particles: np.ndarray = dataset[:, :-1]
        self.a, self.d = sure_particles.shape
        sure_vel: np.ndarray = np.zeros_like(sure_particles)
        sure_states: np.ndarray = np.stack([sure_particles, sure_vel], axis=1)

        self.mins, self.maxs = (
            np.min(sure_particles, axis=0),
            np.max(sure_particles, axis=0),
        )
        del sure_particles, sure_vel

        unsure_particles: np.ndarray = (
            sobol_sequence.sample(int(self.a * unsure_ratio), self.d)
            * (self.maxs - self.mins)
            + self.mins
        )
        self.b = unsure_particles.shape[0]
        vel_mean = (self.maxs - self.mins) / 4
        vel_std = (self.maxs - self.mins) / 16
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
        self.sample_coefs: np.ndarray = np.empty(self.a + self.b)
        self.sample_coefs[: self.a] = 1 / cls_coefs[dataset[:, -1].astype(int)]
        self.sample_coefs[self.a :] = 1

    def state_derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        state = y.reshape(self.a + self.b, 2, self.d)
        prev_pos: np.ndarray = state[:, 0, :].reshape(self.a + self.b, self.d)

        all_pos = np.broadcast_to(prev_pos, (self.b, self.a + self.b, self.d))
        unsure_pos = np.broadcast_to(
            prev_pos[self.a :].reshape(self.b, 1, self.d), (self.b, *prev_pos.shape)
        )
        all_to_unsure = unsure_pos - all_pos
        # dist: (b, a+b)
        dist: np.ndarray = np.linalg.norm(all_to_unsure, axis=2) + UnsureSimulator.EPS
        all_to_unsure /= np.expand_dims(dist, axis=2)
        force_mag = (1 / dist ** 2) * np.broadcast_to(
            self.sample_coefs.reshape(1, -1), (self.b, self.a + self.b)
        )
        all_to_unsure *= (
            np.broadcast_to(
                np.expand_dims(force_mag, axis=2), (*force_mag.shape, self.d)
            )
            * np.broadcast_to(self.maxs - self.mins, (*force_mag.shape, self.d))
            / np.sum(self.sample_coefs)
        )
        # all_to_unsure: (b, d)
        all_to_unsure = np.nansum(all_to_unsure, axis=1)

        unsure_pos = prev_pos[self.a :].clip(self.mins, self.maxs)
        # dist: (b, d)
        dist = unsure_pos - self.mins + UnsureSimulator.EPS
        wall_to_unsure = np.ones_like(dist)
        up_force_mag = (1 / dist ** 2) * UnsureSimulator.WALL_COEF
        dist = self.maxs - unsure_pos + UnsureSimulator.EPS
        down_force_mag = (1 / dist ** 2) * UnsureSimulator.WALL_COEF
        wall_to_unsure *= (up_force_mag - down_force_mag) * np.broadcast_to(
            self.maxs - self.mins, (self.b, self.d)
        )

        unsure_vel = state[self.a :, 1, :]
        friction = -UnsureSimulator.FRIC_COEF * unsure_vel

        delta: np.ndarray = np.empty_like(state)
        delta[:, 0, :] = state[:, 1, :]
        delta[: self.a, 1, :] = 0
        delta[self.a :, 1, :] = all_to_unsure + wall_to_unsure + friction
        return delta.reshape(-1)

    def simulate(self) -> np.ndarray:
        sol = solve_ivp(self.state_derivative, [0, 10], self.state.reshape(-1))
        last_state: np.ndarray = sol.y[:, -1].reshape(self.a + self.b, 2, self.d)
        return last_state[self.a :, 0, :].reshape(self.b, self.d)
