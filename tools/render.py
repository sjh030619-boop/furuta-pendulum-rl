import os, sys, math
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco, numpy as np
from PIL import Image
D = sys.argv[1]; OUT = sys.argv[2]
for variant in ('with_tip', 'no_tip'):
    m = mujoco.MjModel.from_xml_path(os.path.join(D, 'furuta_rev18b_%s.xml' % variant)); d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 1125, 1500)
    for key, q in (('down', [0, 0]), ('up', [0.6, math.pi]), ('swing', [0.0, 2.0])):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        for camname in ('iso', 'front', 'head'):
            if variant == 'no_tip' and camname != 'iso': continue
            r.update_scene(d, camera=camname)
            Image.fromarray(r.render()).save(os.path.join(OUT, 'mj_%s_%s_%s.png' % (variant, key, camname)))
    r.close()
print("ok")
