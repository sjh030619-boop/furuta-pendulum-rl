"""Checks: compile, mass matrix & gravity vs the independent Lagrangian model (furuta_model.py), small-swing period, energy drift."""
import os, sys, json, math
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
from furuta_model import build
mpj = json.load(open(os.path.join(D, 'massprops.json')))
U = np.array(mpj['pend_axis']); piv = np.array(mpj['pivot_m'])
a = math.atan2(U[1], U[0]); Rz = np.array([[math.cos(-a), -math.sin(-a), 0], [math.sin(-a), math.cos(-a), 0], [0, 0, 1]])
report = {}
for variant in ('with_tip', 'no_tip'):
    m = mujoco.MjModel.from_xml_path(os.path.join(D, 'furuta_rev18b_%s.xml' % variant)); d = mujoco.MjData(m)
    r = report[variant] = {}
    r['body_mass_kg'] = {m.body(i).name: round(float(m.body_mass[i]), 5) for i in range(1, m.nbody)}
    # independent model (arm frame rotated so the pivot axis is +x)
    A = mpj['arm']; P = mpj['pend_' + variant]
    Ca = np.array(A['com_m']); Jz = A['inertia_about_com_kgm2'][2][2] + A['mass_kg']*(Ca[0]**2 + Ca[1]**2)
    c0 = Rz @ (np.array(P['com_m']) - piv); Ic = Rz @ np.array(P['inertia_about_com_kgm2']) @ Rz.T
    fM, fh, fE = build(dict(Jz_arm=Jz, m_p=P['mass_kg'], c0=c0.tolist(), Ic=Ic.tolist(), u=[1.0, 0.0, 0.0]))
    errM = errG = 0.0
    rng = np.random.default_rng(0)
    for _ in range(50):
        th, al = rng.uniform(-math.pi, math.pi, 2)
        d.qpos[:] = [th, al]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        M = np.zeros((2, 2))
        try: mujoco.mj_fullM(m, d, M)
        except TypeError: mujoco.mj_fullM(m, M, d.qM)
        Mr = fM(th, al); errM = max(errM, np.abs(M - Mr).max()/np.abs(Mr).max())
        g = d.qfrc_bias.copy(); gr = fh(th, al, 0, 0); errG = max(errG, np.abs(g - gr).max()/(np.abs(gr).max() + 1e-9))
    r['M_down'] = fM(0, 0).tolist(); r['max_rel_err_M'] = errM; r['max_rel_err_gravity'] = errG
    # small-swing period with the arm held (huge armature), no friction
    m.dof_frictionloss[:] = 0; m.dof_damping[:] = 0; m.dof_armature[0] = 1e6
    mujoco.mj_resetData(m, d); d.qpos[:] = [0, math.radians(3)]
    ts, prev, t = [], d.qpos[1], 0.0
    while t < 6.0:
        mujoco.mj_step(m, d); t = d.time
        if prev < 0 <= d.qpos[1]: ts.append(t)
        prev = d.qpos[1]
    T_sim = float(np.mean(np.diff(ts)))
    rp = c0 - np.array([1, 0, 0])*np.dot(c0, [1, 0, 0]); l = np.linalg.norm(rp)
    Jp = Ic[0, 0] + P['mass_kg']*l*l
    T_an = 2*math.pi*math.sqrt(Jp/(P['mass_kg']*9.81*l))
    r['period_sim_s'] = T_sim; r['period_analytic_s'] = T_an; r['pend_inertia_about_axis'] = Jp; r['l_com_m'] = l
    # energy drift, free arm, no friction, start near upright
    m.dof_armature[0] = 0; m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_resetData(m, d); d.qpos[:] = [0, math.radians(170)]; mujoco.mj_forward(m, d)
    E0 = d.energy.sum(); Emax = 0
    for _ in range(5000):
        mujoco.mj_step(m, d); Emax = max(Emax, abs(d.energy.sum() - E0))
    r['energy_drift_J_5s'] = Emax; r['E_scale_J'] = P['mass_kg']*9.81*2*l
    print(variant, json.dumps({k: v for k, v in r.items() if k != 'M_down'}, indent=1))
json.dump(report, open(os.path.join(D, 'check_report.json'), 'w'), indent=1)
