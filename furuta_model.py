"""Furuta pendulum, full 3D inertia Lagrangian (sympy -> numpy).
q = [theta (arm about z), alpha (pendulum about u, 0 = hanging down)]."""
import sympy as sp, numpy as np, json
th, al, thd, ald = sp.symbols('th al thd ald')
def build(P):
    # P: dict with Jz_arm (kg m2, arm group about z incl. motor rotor), m_p (kg), c0 (pendulum COM, m, arm frame at alpha=0, pivot at origin),
    #    Ic (3x3 kg m2 pendulum inertia about its COM, arm frame at alpha=0), u (unit pivot axis in arm frame)
    u = sp.Matrix(P['u']); c0 = sp.Matrix(P['c0']); Ic = sp.Matrix(P['Ic'])
    ux,uy,uz = u
    K = sp.Matrix([[0,-uz,uy],[uz,0,-ux],[-uy,ux,0]])
    Ru = sp.eye(3) + sp.sin(al)*K + (1-sp.cos(al))*K*K
    Rz = sp.Matrix([[sp.cos(th),-sp.sin(th),0],[sp.sin(th),sp.cos(th),0],[0,0,1]])
    R = Rz*Ru
    c = Rz*(Ru*c0)               # pivot axis passes through origin (z axis) by construction
    cdot = c.jacobian([th,al])*sp.Matrix([thd,ald])
    w = sp.Matrix([0,0,thd]) + Rz*u*ald
    Iw = R*Ic*R.T
    T = sp.Rational(1,2)*P['Jz_arm']*thd**2 + sp.Rational(1,2)*P['m_p']*(cdot.T*cdot)[0] + sp.Rational(1,2)*(w.T*Iw*w)[0]
    V = P['m_p']*9.81*c[2]
    qd = sp.Matrix([thd,ald]); q = sp.Matrix([th,al])
    M = sp.hessian(T, qd)
    dTdqd = sp.Matrix([sp.diff(T,v) for v in qd])
    # C*qd term = d/dt(dT/dqd) - M qdd - dT/dq
    Cterm = dTdqd.jacobian(q)*qd - sp.Matrix([sp.diff(T,v) for v in q])
    G = sp.Matrix([sp.diff(V,v) for v in q])
    fM_ = sp.lambdify((th,al), [M[0,0],M[0,1],M[1,1]], modules='math', cse=True)
    fh_ = sp.lambdify((th,al,thd,ald), list(Cterm+G), modules='math', cse=True)
    import numpy as _np
    def f_M(a,b):
        m00,m01,m11 = fM_(a,b); return _np.array([[m00,m01],[m01,m11]])
    def f_h(a,b,c,d): return _np.array(fh_(a,b,c,d))
    f_E = sp.lambdify((th,al,thd,ald), (T,V), modules='math', cse=True)
    f_M.fast = fM_; f_h.fast = fh_
    return f_M, f_h, f_E

def params_from_massprops(mp, Jz_extra=0.0, u_angle_deg=-48.0, pivot_z_mm=14.0):
    a = np.radians(u_angle_deg); u = np.array([np.cos(a), np.sin(a), 0.0])
    # rotate everything so u is +x (arm frame) -> simpler
    Rz = np.array([[np.cos(-a),-np.sin(-a),0],[np.sin(-a),np.cos(-a),0],[0,0,1]])
    pend = mp['pend']; rot = mp['rot']
    C = Rz@np.array(pend['C']); C[2] -= pivot_z_mm
    Ic = Rz@np.array(pend['Ic'])@Rz.T
    return dict(Jz_arm=rot['Io'][2][2]*1e-9 + Jz_extra, m_p=pend['m']*1e-3, c0=(C*1e-3).tolist(), Ic=(Ic*1e-9).tolist(), u=[1.0,0.0,0.0])
