import numpy as np
import torch
from SALib.sample import sobol_sequence as sob_seq
from scipy.integrate import solve_ivp


def normalize_dataset(
    m: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor, bounds: Tuple[float, float]
) -> torch.Tensor:
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


class UnsureSimulator:
    def __init__(
        self,
        dataset: np.ndarray,
        cls_coefs: np.ndarray,
        unsure_ratio: float,
        bounds: Tuple[float, float] = (-10.0, 10.0),
    ):
        sure_particles = torch.from_numpy(dataset[:, :-1]).clone()
        self.a, self.d = sure_particles.shape
        sure_vel: np.ndarray = np.zeros_like(sure_particles)
        sure_states: np.ndarray = np.stack([sure_particles, sure_vel], axis=1)

        self.mins, self.maxs = (
            sure_particles.min(dim=0).values.unsqueeze_(dim=0),
            sure_particles.max(dim=0).values.unsqueeze_(dim=0),
        )
        self.bounds = bounds
        sure_particles = normalize_dataset(sure_particles, self.mins, self.maxs, bounds)
        sure_vel = torch.zeros_like(sure_particles)
        sure_states = torch.stack([sure_particles, sure_vel], dim=1)
        del sure_particles, sure_vel

        unsure_particles: np.ndarray = normalize_dataset(
            torch.from_numpy(sob_seq.sample(int(self.a * unsure_ratio), self.d)),
            torch.tensor(0.0),
            torch.tensor(1.0),
            (
                bounds[0] + (bounds[1] - bounds[0]) / 1000,
                bounds[1] - (bounds[1] - bounds[0]) / 1000,
            ),
        )
        self.b = unsure_particles.shape[0]
        vel_mean = (bounds[1] - bounds[0]) / 4
        vel_std = (bounds[1] - bounds[0]) / 16
        unsure_vel = (
            torch.randn_like(unsure_particles) * vel_std + vel_mean
        ) * torch.tensor([-1, 1])[
            torch.multinomial(
                torch.ones(2) / 2,
                num_samples=unsure_particles.numel(),
                replacement=True,
            )
        ].reshape(
            unsure_particles.shape
        )
        unsure_state = torch.stack([unsure_particles, unsure_vel], dim=1)
        del unsure_particles, unsure_vel

        self.state = torch.cat([sure_states, unsure_state], dim=0)
        self.sample_coefs = torch.empty((1, self.a + self.b))
        self.sample_coefs[0, : self.a] = 1 / torch.from_numpy(
            cls_coefs[dataset[:, -1].astype(int)]
        )
        self.sample_coefs[0, self.a :] = 1
        self.sample_coefs_sum = self.sample_coefs.sum()
        self.empty_like_state = torch.empty_like(self.state)

    def state_derivative(
        self,
        t: float,
        y: np.ndarray,
        EPS: float = 1e-7,
        I2P_COEF: float = 1e0,
        WALL_COEF: float = 1e-1,
        FRIC_COEF: float = 1.0,
    ) -> np.ndarray:
        state = torch.from_numpy(y).reshape(self.a + self.b, 2, self.d)
        prev_pos = state[:, 0, :]

        # all_pos: (1, A+B, D)
        all_pos = prev_pos[:].unsqueeze_(dim=0)
        # unsure_pos: (B, 1, D)
        unsure_pos = prev_pos[self.a :].unsqueeze_(dim=1)
        # all_to_unsure: (B, A+B, D)
        all_to_unsure = unsure_pos - all_pos
        dist = all_to_unsure.square().sum(dim=2).sqrt_() + EPS
        dist[:, self.a :].fill_diagonal_(0)
        all_to_unsure /= dist[:].unsqueeze_(dim=2)
        force_mag = (1 / dist ** 2) * self.sample_coefs * I2P_COEF
        all_to_unsure *= force_mag[:].unsqueeze_(dim=2)
        all_to_unsure /= self.sample_coefs_sum
        all_to_unsure = all_to_unsure.nansum(dim=1)

        unsure_pos = prev_pos[self.a :]
        dist = unsure_pos - self.bounds[0]
        wall_to_unsure = torch.ones_like(dist)
        up_force_mag = (-dist / WALL_COEF).exp()
        dist = self.bounds[1] - unsure_pos
        down_force_mag = (-dist / WALL_COEF).exp()
        wall_to_unsure *= up_force_mag - down_force_mag

        unsure_vel = state[self.a :, 1, :]
        friction = -FRIC_COEF * unsure_vel

        delta = self.empty_like_state
        delta[:, 0, :] = state[:, 1, :]
        delta[: self.a, 1, :] = 0
        delta[self.a :, 1, :] = all_to_unsure + wall_to_unsure + friction
        return delta.reshape(-1).numpy()

    def simulate(self) -> np.ndarray:
        sol = solve_ivp(
            self.state_derivative,
            [0, 10],
            self.state.reshape(-1).numpy(),
            method="RK23",
        )
        last_state: np.ndarray = sol.y[:, -1].reshape(self.a + self.b, 2, self.d)
        return denormalize_dataset(
            last_state[self.a :, 0, :].reshape(self.b, self.d),
            self.mins.numpy(),
            self.maxs.numpy(),
            self.bounds,
        )
