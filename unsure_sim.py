import numpy as np
from SALib.sample import sobol_sequence
from scipy.integrate import solve_ivp


class UnsureSimulator:
    def __init__(self, dataset: np.ndarray, cls_coefs: np.ndarray, unsure_ratio: float):
        sure_particles: np.ndarray = dataset[:, :-1]
        self.a, self.d = sure_particles.shape
        sure_vel: np.ndarray = np.zeros_like(sure_particles)
        sure_states: np.ndarray = np.stack([sure_particles, sure_vel], axis=1)

        self.mins, self.maxs = (
            np.expand_dims(np.min(sure_particles, axis=0), axis=0),
            np.expand_dims(np.max(sure_particles, axis=0), axis=0),
        )
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

    def state_derivative(
        self,
        t: float,
        y: np.ndarray,
        EPS: float = 1e-7,
        I2P_COEF: float = 1e0,
        WALL_COEF: float = 1e-1,
        FRIC_COEF: float = 1.0,
    ) -> np.ndarray:
        state = y.reshape(self.a + self.b, 2, self.d)
        prev_pos: np.ndarray = state[:, 0, :]

        # all_pos: (1, A+B, D)
        all_pos = np.expand_dims(prev_pos, axis=0)
        # unsure_pos: (B, 1, D)
        unsure_pos = np.expand_dims(prev_pos[self.a :], axis=1)
        # all_to_unsure: (B, A+B, D)
        all_to_unsure = unsure_pos - all_pos
        dist: np.ndarray = np.sqrt(np.sum(all_to_unsure ** 2, axis=2)) + EPS
        all_to_unsure /= np.expand_dims(dist, axis=2)
        force_mag = (1 / dist ** 2) * np.expand_dims(self.sample_coefs, axis=0)
        all_to_unsure *= np.expand_dims(force_mag, axis=2)
        all_to_unsure *= np.expand_dims(self.maxs - self.mins, axis=0)
        all_to_unsure /= np.sum(self.sample_coefs)
        all_to_unsure = np.nansum(all_to_unsure, axis=1)

        unsure_pos = prev_pos[self.a :]
        dist = unsure_pos - self.bounds[0]
        wall_to_unsure = np.ones_like(dist)
        up_force_mag = np.exp(-dist / WALL_COEF)
        dist = self.bounds[1] - unsure_pos
        down_force_mag = np.exp(-dist / WALL_COEF)
        wall_to_unsure *= up_force_mag - down_force_mag

        unsure_vel = state[self.a :, 1, :]
        friction = -FRIC_COEF * unsure_vel

        delta: np.ndarray = np.empty_like(state)
        delta[:, 0, :] = state[:, 1, :]
        delta[: self.a, 1, :] = 0
        delta[self.a :, 1, :] = all_to_unsure + wall_to_unsure + friction
        return delta.reshape(-1)

    def simulate(self) -> np.ndarray:
        sol = solve_ivp(
            self.state_derivative, [0, 10], self.state.reshape(-1), method="RK23"
        )
        last_state: np.ndarray = sol.y[:, -1].reshape(self.a + self.b, 2, self.d)
        return last_state[self.a :, 0, :].reshape(self.b, self.d)
