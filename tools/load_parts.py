import sys, pickle, time
sys.path.insert(0, '/home/claude/mj/src')
from xload import load
t=time.time()
P = load(sys.argv[1])
print('parts', len(P), 'in', round(time.time()-t,1), 's')
from OCP.BRepTools import BRepTools
out=[]
for name, shp, rgb in P:
    leaf = name.split('/')[-1]
    out.append((leaf, name, shp.wrapped, rgb))
import OCP
# pickle TopoDS via BRep strings
from OCP.BRepTools import BRepTools
import io
ser=[]
for leaf,name,w,rgb in out:
    import tempfile, os
    fn=tempfile.mktemp(suffix='.brep'); BRepTools.Write_s(w, fn); s=open(fn,'rb').read(); os.remove(fn)
    ser.append((leaf,name,s,rgb))
pickle.dump(ser, open('/home/claude/mj/parts.pkl','wb'))
from collections import Counter
print(Counter(l.rstrip('0123456789').rstrip('_') for l,_,_,_ in ser).most_common(80))
