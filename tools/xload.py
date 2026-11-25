# Load a STEP assembly with names (XCAF) -> list of (name, TopoDS_Shape located in world, color)
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.Quantity import Quantity_Color
from OCP.XCAFDoc import XCAFDoc_ColorType
import cadquery as cq

def load(path):
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    r = STEPCAFControl_Reader(); r.SetNameMode(True); r.SetColorMode(True)
    assert r.ReadFile(path) == 1
    r.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    out = []
    def name_of(lab):
        n = TDataStd_Name()
        if lab.FindAttribute(TDataStd_Name.GetID_s(), n):
            return n.Get().ToExtString()
        return ''
    def walk(lab, loc, path):
        if st.IsAssembly_s(lab):
            comps = TDF_LabelSequence(); st.GetComponents_s(lab, comps)
            for i in range(1, comps.Length()+1):
                c = comps.Value(i)
                ref = TDF_Label(); st.GetReferredShape_s(c, ref)
                cl = st.GetLocation_s(c)
                walk(ref, loc.Multiplied(cl), path + [name_of(c) or name_of(ref)])
        else:
            shp = st.GetShape_s(lab)
            col = Quantity_Color()
            rgb = None
            for ctype in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
                if ct.GetColor_s(lab, ctype, col):
                    rgb = (col.Red(), col.Green(), col.Blue()); break
            out.append(('/'.join(p for p in path if p) or name_of(lab), cq.Shape.cast(shp.Moved(loc)), rgb))
    roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
    for i in range(1, roots.Length()+1):
        walk(roots.Value(i), TopLoc_Location(), [])
    return out
