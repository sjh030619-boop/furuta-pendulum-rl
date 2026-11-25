"""rev18b (square base) STEP -> simplified MuJoCo model.
Three rigid bodies: base (fixed), arm (hinge about z), pendulum (hinge along the -48 deg shaft axis, pivot z = 14 mm).
Mass/COM/inertia are computed from the STEP solids x material density (not from the visual meshes)."""
import pickle, json, math, os, re, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape
from OCP.BRep import BRep_Builder
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/mj/out'
os.makedirs(OUT + '/meshes', exist_ok=True)
ARM_DEG = -48.0; PIVOT_Z = 14.0                          # mm, from the CAD (shaft axis through the motor axis)
U = np.array([math.cos(math.radians(ARM_DEG)), math.sin(math.radians(ARM_DEG)), 0.0])
BEARING_CATALOG_G = 3.6                                   # 688ZZ catalogue mass; CAD is a solid ring (5.9 g)
rows = pickle.load(open('/home/claude/mj/rows.pkl', 'rb'))
ser = {r[1]: r for r in pickle.load(open('/home/claude/mj/parts.pkl', 'rb'))}
for r in rows:
    if r['name'].startswith('688ZZ'):
        k = BEARING_CATALOG_G / r['m']; r['m'] *= k; r['Ic'] = (np.array(r['Ic'])*k).tolist(); r['mat'] = 'steel (688ZZ catalogue mass)'
TIP = ('tip_weight', 'CSK_M3x12_tip')
def combine(rs, about=None):
    m = sum(r['m'] for r in rs); C = sum(np.array(r['C'])*r['m'] for r in rs)/m
    ref = C if about is None else np.array(about)
    I = np.zeros((3, 3))
    for r in rs:
        d = np.array(r['C']) - ref
        I += np.array(r['Ic']) + r['m']*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
    return m, C, I
def shape(s):
    fn = tempfile.mktemp(suffix='.brep'); open(fn, 'wb').write(s)
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, fn, BRep_Builder()); os.remove(fn); return sh
def tess(sh, tol=0.25, ang=0.45):
    BRepMesh_IncrementalMesh(sh, tol, False, ang, True)
    V, F = [], []
    ex = TopExp_Explorer(sh, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current()); loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is not None:
            T = loc.Transformation(); off = len(V)
            for i in range(1, tri.NbNodes()+1):
                p = tri.Node(i).Transformed(T); V.append((p.X(), p.Y(), p.Z()))
            rev = f.Orientation() == 1
            for i in range(1, tri.NbTriangles()+1):
                a, b, c = tri.Triangle(i).Get()
                F.append((off+a-1, off+c-1, off+b-1) if rev else (off+a-1, off+b-1, off+c-1))
        ex.Next()
    return np.array(V), np.array(F, dtype=np.int64)
def write_stl(fn, V, F):
    tri = V[F]; n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]); n /= (np.linalg.norm(n, axis=1)[:, None] + 1e-12)
    rec = np.zeros(len(F), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = tri
    with open(fn, 'wb') as f:
        f.write(b'furuta rev18b simplified'.ljust(80, b' ')); f.write(np.uint32(len(F)).tobytes()); f.write(rec.tobytes())
COL = {'print': (0.82, 0.83, 0.85, 1), 'metal': (0.74, 0.76, 0.79, 1), 'screw_silver': (0.78, 0.79, 0.82, 1), 'screw_black': (0.08, 0.08, 0.09, 1),
       'brass': (0.80, 0.60, 0.22, 1), 'magnet': (0.40, 0.10, 0.30, 1), 'motor': (0.15, 0.15, 0.17, 1), 'cable': (0.75, 0.10, 0.10, 1),
       'dark': (0.12, 0.12, 0.14, 1), 'other': (0.6, 0.6, 0.62, 1)}
def vis_key(r):
    v = r['vis']
    if r['group'] == 'pend' and r['name'].startswith(TIP):
        return 'tip_' + v, COL.get(v, COL['other'])
    if v == 'logo':
        return ('logo_blue', (0.10, 0.35, 0.80, 1)) if 'blue' in r['name'] else ('logo_black', (0.10, 0.10, 0.11, 1))
    if v in ('pcb', 'logo', 'other') and r['rgb']:
        c = tuple(round(x, 2) for x in r['rgb']); return '%s_%02x%02x%02x' % (v, int(c[0]*255), int(c[1]*255), int(c[2]*255)), (*c, 1)
    return v, COL.get(v, COL['other'])
# ---------------------------------------------------------------- meshes per (body, visual class)
geoms = {'base': [], 'arm': [], 'pend': []}
tri_count = 0
for g in geoms:
    buckets = {}
    for r in [x for x in rows if x['group'] == g]:
        k, col = vis_key(r)
        V, F = tess(shape(ser[r['path']][2]), *((0.6, 0.8) if r['vis'] == 'motor' and g == 'base' else (0.25, 0.45)))
        if len(F) == 0: continue
        V = V.copy()
        if g == 'pend': V[:, 2] -= PIVOT_Z
        V *= 1e-3
        b = buckets.setdefault(k, [[], [], col, 0, []])
        F2 = F + b[3]; b[0].append(V); b[1].append(F2); b[3] += len(V); b[4].append(r['name'])
    for k, (Vs, Fs, col, _, names) in buckets.items():
        V = np.vstack(Vs); F = np.vstack(Fs); tri_count += len(F)
        fn = '%s_%s.stl' % (g, k); write_stl(os.path.join(OUT, 'meshes', fn), V, F)
        geoms[g].append((fn, col, names, 'tip' if k.startswith('tip_') else ''))
# tip weight parts get their own geoms so the no-weight variant can drop them
print('triangles', tri_count)
# ---------------------------------------------------------------- inertial
mp = {}
arm_rows = [r for r in rows if r['group'] == 'arm']
m, C, I = combine(arm_rows); mp['arm'] = dict(m=m, C=C, I=I)
for variant, prs in (('with_tip', [r for r in rows if r['group'] == 'pend']),
                     ('no_tip', [r for r in rows if r['group'] == 'pend' and not r['name'].startswith(TIP)])):
    m, C, I = combine(prs); mp['pend_' + variant] = dict(m=m, C=C, I=I)
def fullinertia(I):  # g mm^2 -> kg m^2, order ixx iyy izz ixy ixz iyz
    I = I*1e-9; return '%.6e %.6e %.6e %.6e %.6e %.6e' % (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
def vec(v, s=1e-3): return ' '.join('%.6f' % (x*s) for x in v)
def mjcf(variant):
    P = mp['pend_' + variant]; A = mp['arm']
    pc = P['C'] - np.array([0, 0, PIVOT_Z])
    L = []
    w = L.append
    w('<mujoco model="furuta_rev18b_%s">' % variant)
    w('  <!-- generated by build_mjcf.py from 4_Archives/rev18b/01_CAD/Furuta_rev18b_full.step; units m, kg, s -->')
    w('  <compiler angle="radian" meshdir="meshes" inertiafromgeom="false" autolimits="true"/>')
    w('  <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81"/>')
    w('  <visual><headlight ambient="0.22 0.22 0.22" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/><global offwidth="2000" offheight="1500"/>'
      '<quality shadowsize="4096"/></visual>')
    w('  <default>')
    w('    <geom contype="0" conaffinity="0" group="1" density="0"/>')
    w('    <default class="arm_joint"><joint damping="1e-4" frictionloss="0.012" armature="0"/></default>')
    w('    <default class="pend_joint"><joint damping="1e-5" frictionloss="2e-4" armature="0"/></default>')
    w('  </default>')
    w('  <asset>')
    for g in geoms:
        for fn, col, names, tag in geoms[g]:
            if variant == 'no_tip' and tag == 'tip': continue
            w('    <mesh name="%s" file="%s"/>' % (fn[:-4], fn))
    w('    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.92 0.93 0.95" width="256" height="256"/>')
    w('    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.94 0.94" rgb2="0.88 0.88 0.88" width="512" height="512"/>')
    w('    <material name="floor" texture="grid" texrepeat="8 8"/>')
    w('  </asset>')
    w('  <worldbody>')
    w('    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.45 0.45 0.45"/>')
    w('    <light pos="-0.4 -0.2 0.5" dir="0.4 0.2 -0.5" diffuse="0.18 0.18 0.18" castshadow="false"/>')
    w('    <geom name="floor" type="plane" size="0.4 0.4 0.01" pos="0 0 -0.2082" material="floor" group="0"/>')
    w('    <camera name="iso" pos="-0.2343 -0.6438 0.2151" xyaxes="0.9397 -0.3420 0.0000 0.1391 0.3822 0.9135"/>')
    w('    <camera name="front" pos="-0.5560 -0.5006 -0.0377" xyaxes="0.6691 -0.7431 0.0000 0.0518 0.0467 0.9976"/>')
    w('    <camera name="head" pos="-0.1099 -0.2450 0.1500" xyaxes="0.8660 -0.5000 0.0000 0.2500 0.4330 0.8660"/>')
    w('    <!-- base: tower, top plate, motor stator/module, electronics box (fixed) -->')
    for fn, col, names, tag in geoms['base']:
        w('    <geom type="mesh" mesh="%s" rgba="%.3f %.3f %.3f 1"/>' % (fn[:-4], *col[:3]))
    w('    <body name="arm" pos="0 0 0">')
    w('      <!-- rotating head: drum, lid, bearing cartridge, AS5600 board, hub tube, motor output -->')
    w('      <inertial pos="%s" mass="%.6f" fullinertia="%s"/>' % (vec(A['C']), A['m']*1e-3, fullinertia(A['I'])))
    w('      <joint name="arm" class="arm_joint" type="hinge" axis="0 0 1"/>')
    for fn, col, names, tag in geoms['arm']:
        w('      <geom type="mesh" mesh="%s" rgba="%.3f %.3f %.3f 1"/>' % (fn[:-4], *col[:3]))
    w('      <site name="encoder_chip" pos="-0.0090 0.0100 0.0140" size="0.001"/>')
    w('      <body name="pendulum" pos="0 0 %.6f">' % (PIVOT_Z*1e-3))
    w('        <!-- SUS shaft + magnet holder + pendulum hub + SUS rod%s; angle 0 = hanging down -->' % (' + tip weight' if variant == 'with_tip' else ''))
    w('        <inertial pos="%s" mass="%.6f" fullinertia="%s"/>' % (vec(pc), P['m']*1e-3, fullinertia(P['I'])))
    w('        <joint name="pendulum" class="pend_joint" type="hinge" axis="%.6f %.6f 0"/>' % (U[0], U[1]))
    for fn, col, names, tag in geoms['pend']:
        if variant == 'no_tip' and tag == 'tip': continue
        w('        <geom type="mesh" mesh="%s" rgba="%.3f %.3f %.3f 1"/>' % (fn[:-4], *col[:3]))
    tip = np.array([80*U[0], 80*U[1], -126.0 - PIVOT_Z])*1e-3
    w('        <site name="tip" pos="%s" size="0.002"/>' % ' '.join('%.5f' % x for x in tip))
    w('      </body>')
    w('    </body>')
    w('  </worldbody>')
    w('  <actuator>')
    w('    <!-- arm torque (N m). RMD-X6-P20 can do far more; keep this clamp for training safety and widen deliberately -->')
    w('    <motor name="arm_torque" joint="arm" gear="1" ctrlrange="-1 1"/>')
    w('  </actuator>')
    w('  <sensor>')
    w('    <jointpos name="arm_angle" joint="arm"/><jointpos name="pend_angle" joint="pendulum"/>')
    w('    <jointvel name="arm_vel" joint="arm"/><jointvel name="pend_vel" joint="pendulum"/>')
    w('    <actuatorfrc name="arm_torque_applied" actuator="arm_torque"/>')
    w('  </sensor>')
    w('  <keyframe>')
    w('    <key name="down" qpos="0 0"/>')
    w('    <key name="up" qpos="0 3.14159265"/>')
    w('  </keyframe>')
    w('</mujoco>')
    open(os.path.join(OUT, 'furuta_rev18b_%s.xml' % variant), 'w').write('\n'.join(L) + '\n')
mjcf('with_tip'); mjcf('no_tip')
def j(x): return np.asarray(x).tolist()
json.dump({k: dict(mass_kg=v['m']*1e-3, com_m=j(v['C']*1e-3), inertia_about_com_kgm2=j(v['I']*1e-9)) for k, v in mp.items()} |
          {'pivot_m': [0, 0, PIVOT_Z*1e-3], 'pend_axis': j(U), 'parts': [dict(name=r['name'], body=r['group'], material=r['mat'], mass_g=round(r['m'], 3)) for r in rows]},
          open(os.path.join(OUT, 'massprops.json'), 'w'), indent=1, ensure_ascii=False)
for k, v in mp.items(): print(k, 'mass %.1f g' % v['m'], 'COM mm', np.round(v['C'], 2))
