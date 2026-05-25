from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .instance import ProbJobShopInstance
from .simulator import PTASimulator
from .benchmarks.all_instances import get_instance_by_name

if TYPE_CHECKING:
    pass


class JobShopEnv(gym.Env):
    """Gymnasium environment wrapping the PTA job-shop simulator.

    Observation:
        task_status      (N,)  int32   0=waiting 1=active 2=done
        time_remaining   (N,)  float32 duration_if_active - clock; 0 otherwise
        expected_duration(N,)  float32 fixed E[d] per task
        clock_values     (J,)  float32 per-job elapsed since last task start
        current_time     (1,)  float32 global simulation time
        action_mask      (N,)  int8    1 if task currently enabled

    Action:
        Discrete(N) — index into the flat task list (jobs in order, tasks in order).
        Invalid actions raise ValueError.

    Reward modes:
        "sparse"  0 each step; -makespan at terminal
        "dense"   -dt at each internal advance_time call
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: ProbJobShopInstance,
        seed: Optional[int] = None,
        reward_mode: str = "sparse",
    ):
        super().__init__()
        self.instance = instance
        self._seed = seed
        self.reward_mode = reward_mode

        self._task_ids = [t.task_id for j in instance.jobs for t in j.tasks]
        self._task_to_idx = {tid: i for i, tid in enumerate(self._task_ids)}

        n = instance.num_tasks()
        j = len(instance.jobs)
        ub = float(instance.worst_case_makespan_upper_bound())
        max_dur = float(max(t.distribution.max_duration() for t in instance.all_tasks()))

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict({
            "task_status":       spaces.Box(0, 2,   shape=(n,), dtype=np.int32),
            "time_remaining":    spaces.Box(0.0, max_dur, shape=(n,), dtype=np.float32),
            "expected_duration": spaces.Box(0.0, max_dur, shape=(n,), dtype=np.float32),
            "clock_values":      spaces.Box(0.0, ub,      shape=(j,), dtype=np.float32),
            "current_time":      spaces.Box(0.0, ub,      shape=(1,), dtype=np.float32),
            "action_mask":       spaces.Box(0, 1,   shape=(n,), dtype=np.int8),
        })

        self._expected_durations = np.array(
            [instance.task_by_id(tid).distribution.expected_duration()
             for tid in self._task_ids],
            dtype=np.float32,
        )

        self._sim: Optional[PTASimulator] = None
        self._state = None

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self._sim = PTASimulator(self.instance, seed=effective_seed)
        self._state = self._sim.initial_state()
        return self._get_obs(), {}

    def step(self, action: int):
        task_id = self._task_ids[action]
        enabled = self._state.enabled_tasks(self.instance)
        if task_id not in enabled:
            raise ValueError(
                f"Action {action} ({task_id!r}) is not currently enabled. "
                f"Enabled: {enabled}"
            )

        self._state = self._sim.start_task(self._state, task_id)

        reward = 0.0
        while not self._state.is_terminal() and not self._state.enabled_tasks(self.instance):
            prev = self._state.current_time
            self._state = self._sim.advance_time(self._state)
            if self.reward_mode == "dense":
                reward -= (self._state.current_time - prev)

        terminated = self._state.is_terminal()
        if terminated and self.reward_mode == "sparse":
            reward = -self._state.current_time

        return self._get_obs(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        return self._get_obs()["action_mask"]

    def _action_mask(self) -> np.ndarray:
        enabled = set(self._state.enabled_tasks(self.instance))
        return np.array(
            [1 if tid in enabled else 0 for tid in self._task_ids],
            dtype=np.int8,
        )

    def _get_obs(self) -> dict:
        n = self.instance.num_tasks()
        task_status    = np.zeros(n, dtype=np.int32)
        time_remaining = np.zeros(n, dtype=np.float32)

        for i, tid in enumerate(self._task_ids):
            ts = self._state.task_states[tid]
            if ts.status == "active":
                task_status[i] = 1
                job_id = self.instance.job_of_task(tid).job_id
                time_remaining[i] = max(
                    0.0,
                    ts.duration_if_active - self._state.clock_values[job_id],
                )
            elif ts.status == "done":
                task_status[i] = 2

        clock_values = np.array(
            [self._state.clock_values[job.job_id] for job in self.instance.jobs],
            dtype=np.float32,
        )
        current_time = np.array([self._state.current_time], dtype=np.float32)

        return {
            "task_status":        task_status,
            "time_remaining":     time_remaining,
            "expected_duration":  self._expected_durations.copy(),
            "clock_values":       clock_values,
            "current_time":       current_time,
            "action_mask":        self._action_mask(),
        }


def make_env(instance_name: str, **kwargs) -> JobShopEnv:
    return JobShopEnv(get_instance_by_name(instance_name), **kwargs)
