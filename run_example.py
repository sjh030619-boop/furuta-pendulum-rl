"""Minimal usage: passive release near upright, then a torque pulse on the arm. Saves a plot.
python run_example.py [with_tip|no_tip]"""
import os, sys, math
import numpy as np, mujoco
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
variant = sys.argv[1] if len(sys.argv) > 1 else 'with_tip'
m = mujoco.MjModel.from_xml_path(os.path.join(D, 'furuta_rev18b_%s.xml' % variant)); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key('up').id); d.qpos[:] = [0.0, math.pi - 0.05]   # 3 deg off upright
log = []
while d.time < 6.0:
    d.ctrl[0] = 0.03 if 3.0 <= d.time < 3.2 else 0.0                                     # 0.03 N m pulse on the arm
    mujoco.mj_step(m, d)
    s = d.sensordata                                                                     # arm_angle, pend_angle, arm_vel, pend_vel, torque
    log.append((d.time, s[0], s[1], s[2], s[3], s[4]))
L = np.array(log)
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(L[:, 0], np.degrees(L[:, 2])); ax[0].set_ylabel('pendulum [deg]\n(0 = down, 180 = up)')
ax[1].plot(L[:, 0], np.degrees(L[:, 1])); ax[1].set_ylabel('arm [deg]')
ax[2].plot(L[:, 0], L[:, 5]); ax[2].set_ylabel('arm torque [N m]'); ax[2].set_xlabel('time [s]')
for a in ax: a.grid(alpha=0.3)
fig.suptitle('Furuta rev18b MuJoCo (%s): passive fall from 3 deg off upright, 0.03 N m pulse at 3 s' % variant)
fig.tight_layout(); fig.savefig(os.path.join(D, 'example_%s.png' % variant), dpi=110)
print('final pendulum %.1f deg, arm %.1f deg' % (math.degrees(L[-1, 2]), math.degrees(L[-1, 1])))
