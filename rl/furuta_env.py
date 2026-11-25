"""Furuta pendulum swing-up + balance environment (Gymnasium) on the rev18b MuJoCo model.

State convention follows the MJCF: qpos = [theta (arm, z hinge), alpha (pendulum)],
alpha = 0 hanging down, alpha = pi upright.

Observation (6):  cos(theta), sin(theta), cos(alpha), sin(alpha), theta_dot/W_ARM, alpha_dot/W_PEND
Action (1):       normalized arm torque in [-1, 1] -> tau_max [N m]

The arm joint armature (motor reflected inertia) is set at construction time instead of in
the XML, so the shipped model files stay untouched.

Reward, with phi = alpha - pi wrapped to (-pi, pi] (0 = upright):

    r = 1 - 0.3 phi^2 - 0.002 alpha_dot^2 - 0.002 theta_dot^2 - 0.02 u^2
          + 0.5 exp(-(phi/0.3)^2) exp(-(alpha_dot/2)^2)

Every term is bounded and there is no early termination, so the per-step values form a
monotone ladder -- hanging still ~0.01, fast rotation ~0.3, separatrix orbit ~0.7,
balanced 1.5 -- and each improvement in behaviour is rewarded. Half of the training
episodes start from a uniformly random state (exploring starts) so the balance skill is
seen early; evaluation always starts from hanging.
"""
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

W_ARM = 20.0
W_PEND = 40.0
TWO_PI = 2.0 * math.pi

DEFAULTS = dict(
    xml="furuta_rev18b_no_tip.xml",
    arm_armature=5.0e-3,
    pend_armature=0.0,
    timestep=0.002,
    solver_iterations=0,
    ctrl_hz=50.0,
    episode_sec=5.0,
    tau_max=1.0,
    arm_no_load_speed=16.0,
    task="swingup",
    init_alpha_spread=0.05,
    init_vel_spread=0.05,
    explore_start_frac=0.5,
    obs_noise_deg=0.0,
    action_delay_steps=0,
    vel_limit_arm=200.0,
    vel_limit_pend=200.0,
)


class FurutaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kw):
        cfg = dict(DEFAULTS)
        cfg.update(kw)
        self.cfg = cfg

        path = cfg["xml"] if os.path.isabs(cfg["xml"]) else os.path.join(ROOT, cfg["xml"])
        self.model = mujoco.MjModel.from_xml_path(path)
        if cfg["timestep"] > 0.0:
            self.model.opt.timestep = cfg["timestep"]
        if cfg["solver_iterations"] > 0:
            self.model.opt.iterations = int(cfg["solver_iterations"])
        self.data = mujoco.MjData(self.model)

        self.arm_dof = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm")]
        self.pend_dof = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum")]
        self.model.dof_armature[self.arm_dof] = cfg["arm_armature"]
        self.model.dof_armature[self.pend_dof] = cfg["pend_armature"]

        self.decim = max(1, int(round(1.0 / (cfg["ctrl_hz"] * self.model.opt.timestep))))
        self.ctrl_dt = self.decim * self.model.opt.timestep
        self.max_steps = int(round(cfg["episode_sec"] / self.ctrl_dt))
        self.tau_max = float(cfg["tau_max"])
        self.w0 = float(cfg["arm_no_load_speed"])
        self.task = cfg["task"]
        self.obs_noise = math.radians(cfg["obs_noise_deg"])
        self.delay = int(cfg["action_delay_steps"])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._step = 0
        self._queue = [0.0] * self.delay

        self._buf = np.zeros(6, dtype=np.float32)
        self._mj_step = mujoco.mj_step
        self._qpos = self.data.qpos
        self._qvel = self.data.qvel
        self._ctrl = self.data.ctrl
        self._lim_arm = float(cfg["vel_limit_arm"])
        self._lim_pend = float(cfg["vel_limit_pend"])
        self._info_up = {"upright": 0.0}

    def _motor(self, u, w):
        """First-order speed-torque curve: tau <= tau_stall*(1 - |w|/w_no_load) when driving,
        full tau_max when braking. Gives the actuator a hard speed ceiling, as a real geared
        servo has, instead of unbounded acceleration."""
        tau = u * self.tau_max
        if self.w0 <= 0.0:
            return tau
        if tau * w > 0.0:
            lim = self.tau_max * max(0.0, 1.0 - abs(w) / self.w0)
            return max(-lim, min(lim, tau))
        return tau

    def _obs(self):
        q = self._qpos
        v = self._qvel
        th = q[0]
        al = q[1]
        if self.obs_noise > 0.0:
            th += self.np_random.normal(0.0, self.obs_noise)
            al += self.np_random.normal(0.0, self.obs_noise)
        b = self._buf
        b[0] = math.cos(th)
        b[1] = math.sin(th)
        b[2] = math.cos(al)
        b[3] = math.sin(al)
        b[4] = v[0] / W_ARM
        b[5] = v[1] / W_PEND
        return b.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        c = self.cfg
        if self.task == "balance":
            al0 = math.pi + self.np_random.uniform(-0.25, 0.25)
            th0 = self.np_random.uniform(-0.25, 0.25)
            vel = self.np_random.uniform(-0.5, 0.5, size=2)
        elif self.np_random.random() < c["explore_start_frac"]:
            al0 = self.np_random.uniform(-math.pi, math.pi)
            th0 = self.np_random.uniform(-math.pi, math.pi)
            vel = self.np_random.uniform(-1.0, 1.0, size=2) * np.array([2.0, 6.0])
        else:
            al0 = self.np_random.uniform(-c["init_alpha_spread"], c["init_alpha_spread"])
            th0 = self.np_random.uniform(-c["init_alpha_spread"], c["init_alpha_spread"])
            vel = self.np_random.uniform(-c["init_vel_spread"], c["init_vel_spread"], size=2)
        self.data.qpos[:] = [th0, al0]
        self.data.qvel[:] = vel
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._queue = [0.0] * self.delay
        return self._obs(), {}

    def set_state(self, theta, alpha, theta_dot=0.0, alpha_dot=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [theta, alpha]
        self.data.qvel[:] = [theta_dot, alpha_dot]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._queue = [0.0] * self.delay
        return self._obs()

    def step(self, action):
        u = action[0]
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0
        q = self._qpos
        v = self._qvel
        if self.delay:
            self._queue.append(u)
            applied = self._queue.pop(0)
        else:
            applied = u
        w = v[0]
        tau = applied * self.tau_max
        if self.w0 > 0.0 and tau * w > 0.0:
            lim = self.tau_max * (1.0 - abs(w) / self.w0)
            if lim < 0.0:
                lim = 0.0
            if tau > lim:
                tau = lim
            elif tau < -lim:
                tau = -lim
        self._ctrl[0] = tau
        self._mj_step(self.model, self.data, nstep=self.decim)

        al = q[1]
        thd = v[0]
        ald = v[1]

        phi = (al - math.pi) % TWO_PI
        if phi > math.pi:
            phi -= TWO_PI
        near = math.exp(-11.111111 * phi * phi)
        slow = math.exp(-0.25 * ald * ald)
        r = (1.0 - 0.3 * phi * phi - 0.002 * ald * ald - 0.002 * thd * thd
             - 0.02 * applied * applied + 0.5 * near * slow)

        self._step += 1
        terminated = bool(thd > self._lim_arm or thd < -self._lim_arm
                          or ald > self._lim_pend or ald < -self._lim_pend)
        truncated = self._step >= self.max_steps
        up = 1.0 if (-0.1745 < phi < 0.1745 and -2.0 < ald < 2.0) else 0.0
        return self._obs(), r, terminated, truncated, {"upright": up}


def make_env(**kw):
    def _f():
        return FurutaEnv(**kw)
    return _f
