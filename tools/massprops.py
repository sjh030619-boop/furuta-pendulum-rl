import pickle, json, re, math, sys, os, tempfile
import numpy as np
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape
from OCP.BRep import BRep_Builder
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
ser = pickle.load(open('/home/claude/mj/parts.pkl','rb'))
def shape(s):
    fn = tempfile.mktemp(suffix='.brep'); open(fn,'wb').write(s)
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, fn, BRep_Builder()); os.remove(fn); return sh
# ---- grouping (rev18b names)
PEND = ('shaft_SUS304', 'magnet_holder', 'magnet_8x2', 'Ering', 'shim', 'pendulum_hub', 'CSK_M3x16_pend', 'INS_pend_hub',
        'pendulum_rod', 'tip_weight', 'CSK_M3x12_tip')
ARM = ('drum_rev16d', 'lid_rev16d', 'CSK_M3x8_lid_1', 'CSK_M3x8_lid_2', 'CSK_M3x8_lid_3', 'CSK_M3x8_lid_4', 'INS_drum_lid',
       'bearing_cartridge', '688ZZ', 'Grove', 'AS5600_chip', 'SHCS_M3x8_cartridge', 'INS_drum_cradle', 'hub_rev16g',
       'SHCS_M4x16_drum_hub_motor', 'RMD-X6-P20-60-E_output', 'slipring_drive_adapter', 'cable_AS5600')
# ---- density g/mm^3 and visual class
def material(n):
    if re.search(r'rev1\d', n) and 'logo' not in n: return 'PLA', 1.24e-3, 'print'
    if 'logo' in n: return 'PLA', 1.24e-3, 'logo'
    if n.startswith(('shaft_SUS', 'pendulum_rod_SUS')): return 'SUS304', 7.93e-3, 'metal'
    if n.startswith('magnet_holder_Al'): return 'Al6061', 2.70e-3, 'metal'
    if n.startswith('magnet_8x2'): return 'NdFeB', 7.50e-3, 'magnet'
    if n.startswith(('INS_', 'STANDOFF', 'tip_weight')): return 'brass', 8.50e-3, 'brass'
    if n.startswith(('688ZZ', 'Ering', 'shim')): return 'steel', 7.85e-3, 'metal'
    if n.startswith(('CSK_', 'SHCS_', 'BTN_')): return 'steel', 7.85e-3, 'screw'
    if n.startswith('RMD-X6'): return 'Al(motor)', 2.70e-3, 'motor'
    if n.startswith(('Grove_AS5600_PCB', 'Raspberry', 'SEENGREAT')): return 'PCB', 1.85e-3, 'pcb'
    if n.startswith(('Grove', 'AS5600_chip')): return 'plastic', 1.40e-3, 'pcb'
    if n.startswith('cable'): return 'cable', 2.00e-3, 'cable'
    if n.startswith('FOOT'): return 'silicone', 1.20e-3, 'dark'
    if n.startswith('slipring_SZH'): return 'slipring', 2.50e-3, 'dark'
    return 'misc', 1.50e-3, 'other'
rows = []
for leaf, path, s, rgb in ser:
    n = leaf.split(':')[0]
    if n.startswith('=>'):           # unnamed sub-shapes: take the named parent from the path
        n = [p for p in path.split('/') if not p.startswith('=>')][-1].split(':')[0]
    sh = shape(s)
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(sh, g)
    V = g.Mass()
    if V <= 0: print('zero volume', n); continue
    c = g.CentreOfMass(); I = g.MatrixOfInertia()
    Ic = np.array([[I.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])   # about COM, per unit density (mm^5)
    grp = 'pend' if n.startswith(PEND) else ('arm' if n.startswith(ARM) else 'base')
    mat, rho, vis = material(n)
    # screws: model colour decides black vs silver
    if vis == 'screw':
        vis = 'screw_black' if (rgb and max(rgb) < 0.2) else 'screw_silver'
    rows.append(dict(name=n, path=path, group=grp, mat=mat, rho=rho, vis=vis, V=V, m=V*rho, C=[c.X(), c.Y(), c.Z()], Ic=(Ic*rho).tolist(), rgb=rgb))
pickle.dump(rows, open('/home/claude/mj/rows.pkl', 'wb'))
def combine(rs):
    m = sum(r['m'] for r in rs); C = sum(np.array(r['C'])*r['m'] for r in rs)/m
    I = np.zeros((3, 3))
    for r in rs:
        d = np.array(r['C']) - C
        I += np.array(r['Ic']) + r['m']*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
    return m, C, I
res = {}
for g in ('base', 'arm', 'pend'):
    rs = [r for r in rows if r['group'] == g]
    m, C, I = combine(rs)
    res[g] = dict(n=len(rs), m_g=m, C_mm=C.tolist(), Ic_gmm2=I.tolist())
    print('%-5s parts %3d  mass %.1f g  COM %s' % (g, len(rs), m, np.round(C, 2)))
for r in rows:
    if r['group'] != 'base': print('  %-5s %-32s %-9s %7.2f g  %s' % (r['group'], r['name'], r['mat'], r['m'], r['vis']))
json.dump(res, open('/home/claude/mj/massprops_raw.json', 'w'), indent=1)
