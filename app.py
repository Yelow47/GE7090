import streamlit as st
import numpy as np
import tempfile
import zipfile
import shutil
import os
import struct
import requests
from pathlib import Path
from io import BytesIO
import xml.etree.ElementTree as ET

st.set_page_config(page_title="SARDetect", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#080e1f;color:#e8edf5;font-size:15px;line-height:1.6;}
.stApp{background:radial-gradient(ellipse at 0% 0%,rgba(240,165,0,0.04) 0%,transparent 50%),radial-gradient(ellipse at 100% 100%,rgba(0,200,240,0.04) 0%,transparent 50%),#080e1f;}
h1,h2,h3{font-family:'Share Tech Mono',monospace!important;color:#e8edf5!important}
p,li,span{color:#e8edf5}
.main-title{font-family:'Share Tech Mono',monospace;font-size:2.6rem;color:#f0a500!important;letter-spacing:.1em;text-shadow:0 0 40px rgba(240,165,0,0.3);margin:0;}
.sub-title{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#3d4f6a;letter-spacing:.25em;margin:0 0 1.5rem;}
.card{background:#0c1428;border:1px solid #111e35;border-top:2px solid;border-radius:4px;padding:16px 20px;height:100%;}
.card h4{font-family:'Share Tech Mono',monospace;font-size:.8rem;letter-spacing:.15em;margin:0 0 8px;}
.card p{font-size:.86rem;color:#8a9ab8;margin:0;line-height:1.6;}
.cfar-top{border-top-color:#00c8f0}.cfar-top h4{color:#00c8f0}
.yolo-top{border-top-color:#ff4a5a}.yolo-top h4{color:#ff4a5a}
.comb-top{border-top-color:#4dff91}.comb-top h4{color:#4dff91}
.mbox{background:#0c1428;border:1px solid #111e35;border-radius:4px;padding:20px;text-align:center;}
.mval{font-family:'Share Tech Mono',monospace;font-size:2.6rem;margin:0;line-height:1;}
.mlbl{font-size:.7rem;color:#8a9ab8;letter-spacing:.18em;margin:6px 0 0;text-transform:uppercase;}
.cc{color:#00c8f0}.yc{color:#ff4a5a}.dc{color:#8a9ab8}
.faq-q{font-family:'Share Tech Mono',monospace;color:#f0a500;font-size:.88rem;margin:20px 0 6px;}
.faq-a{color:#8a9ab8;font-size:.88rem;line-height:1.7;margin:0;}
.stButton>button{font-family:'Share Tech Mono',monospace!important;background:#f0a500!important;color:#080e1f!important;border:none!important;border-radius:3px!important;font-weight:700!important;letter-spacing:.1em!important;padding:.55rem 2rem!important;transition:all .2s!important;}
.stButton>button:hover{background:#d49000!important;box-shadow:0 0 24px rgba(240,165,0,0.3)!important;}
div[data-testid="stFileUploader"]{background:#0a1122!important;border:1px dashed #1a2e50!important;border-radius:4px!important;}
section[data-testid="stSidebar"]{background:#0c1428;border-right:1px solid #111e35;}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stMarkdown{color:#e8edf5!important;}
.stSlider>div>div>div{background:#f0a500!important}.stProgress>div>div{background:#f0a500!important}
.stAlert{background:#0c1428!important;border-color:#1a2e50!important;color:#e8edf5!important}
.stCode{background:#0a1122!important}
hr{border-color:#111e35!important;margin:1.5rem 0!important}
.stSelectbox>div>div{background:#0a1122!important;color:#e8edf5!important;border-color:#1a2e50!important}
.upload-hint{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#3d4f6a;letter-spacing:.1em;margin:4px 0 12px;}
.section-header{font-family:'Share Tech Mono',monospace;font-size:.9rem;color:#f0a500;letter-spacing:.15em;margin:1.5rem 0 .75rem;padding-bottom:6px;border-bottom:1px solid #111e35;}
</style>""", unsafe_allow_html=True)

WEIGHTS_URL  = "https://github.com/Yelow47/GE7090/releases/download/SAR/best.pt"
WEIGHTS_PATH = Path("weights/best.pt")
LAND_ZIP     = Path("LandPolygon.zip")
TEMP_DIR     = Path(tempfile.gettempdir()) / "sardetect"
TEMP_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  LAND MASK
# ─────────────────────────────────────────────────────────────

def _read_shp_polygons(shp_path):
    """Parse polygon shapefile without external geo libs. Returns [(bbox, [(x,y)...]),...]"""
    polygons = []
    with open(shp_path, "rb") as f:
        f.seek(100)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            content_len = struct.unpack(">i", hdr[4:])[0]
            content     = f.read(content_len * 2)
            shape_type  = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15, 25):
                continue
            bbox       = struct.unpack("<4d", content[4:36])
            num_parts  = struct.unpack("<i",  content[36:40])[0]
            num_points = struct.unpack("<i",  content[40:44])[0]
            pts_data   = content[44 + num_parts*4 : 44 + num_parts*4 + num_points*16]
            pts_flat   = struct.unpack(f"<{num_points*2}d", pts_data)
            polygons.append((bbox, list(zip(pts_flat[0::2], pts_flat[1::2]))))
    return polygons


@st.cache_data(show_spinner=False)
def _load_land_polygons():
    if not LAND_ZIP.exists():
        return []
    land_dir = TEMP_DIR / "land"
    land_dir.mkdir(exist_ok=True)
    shp_out  = land_dir / "LandPolygon.shp"
    if not shp_out.exists():
        with zipfile.ZipFile(LAND_ZIP) as z:
            z.extractall(land_dir)
    return _read_shp_polygons(str(shp_out))


def _merc_to_geo(x, y):
    """Web Mercator (EPSG:3857) metres → WGS84 degrees."""
    R   = 6378137.0
    lon = np.degrees(np.asarray(x, float) / R)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y, float) / R)) - np.pi / 2)
    return lon, lat


def _utm_to_geo(easting, northing, zone, hemi):
    """Approximate UTM → WGS84 lon/lat (sufficient for masking)."""
    a   = 6378137.0; f = 1/298.257223563; b = a*(1-f)
    e2  = 1-(b/a)**2; e = np.sqrt(e2); k0 = 0.9996
    E0  = 500000.0; N0 = 0.0 if hemi == "N" else 10000000.0
    x   = np.asarray(easting, float) - E0
    y   = np.asarray(northing, float) - N0
    M   = y/k0
    mu  = M/(a*(1-e2/4-3*e2**2/64-5*e2**3/256))
    e1  = (1-np.sqrt(1-e2))/(1+np.sqrt(1-e2))
    p1  = mu+(3*e1/2-27*e1**3/32)*np.sin(2*mu)+(21*e1**2/16-55*e1**4/32)*np.sin(4*mu)+(151*e1**3/96)*np.sin(6*mu)
    N1  = a/np.sqrt(1-e2*np.sin(p1)**2)
    T1  = np.tan(p1)**2; C1=e2/(1-e2)*np.cos(p1)**2
    R1  = a*(1-e2)/(1-e2*np.sin(p1)**2)**1.5
    D   = x/(N1*k0)
    lat = p1-(N1*np.tan(p1)/R1)*(D**2/2-(5+3*T1+10*C1-4*C1**2-9*e2/(1-e2))*D**4/24)
    lon0= np.radians((zone-1)*6-180+3)
    lon = lon0+(D-(1+2*T1+C1)*D**3/6)/np.cos(p1)
    return np.degrees(lon), np.degrees(lat)


def _point_in_polygon(px, py, polygon):
    n=len(polygon); ins=False; j=n-1
    for i in range(n):
        xi,yi=polygon[i]; xj,yj=polygon[j]
        if ((yi>py)!=(yj>py)) and (px<(xj-xi)*(py-yi)/(yj-yi+1e-15)+xi):
            ins=not ins
        j=i
    return ins


def build_land_mask(transform, shape, polygons, crs_epsg):
    """Rasterise Web-Mercator land polygons onto the SAR image grid."""
    from scipy.ndimage import binary_dilation
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    if not polygons:
        return mask

    col_idx = np.arange(cols); row_idx = np.arange(rows)
    px_x = transform.c + col_idx * transform.a   # map x per column
    px_y = transform.f + row_idx * transform.e   # map y per row

    epsg = int(crs_epsg) if crs_epsg else 4326
    if epsg == 4326:
        lon_col = px_x; lat_row = px_y
    elif 32601 <= epsg <= 32660:
        lon_col, lat_row = _utm_to_geo(px_x, px_y, epsg % 100, "N")
    elif 32701 <= epsg <= 32760:
        lon_col, lat_row = _utm_to_geo(px_x, px_y, epsg % 100, "S")
    else:
        lon_col = px_x; lat_row = px_y

    img_lon_min,img_lon_max = lon_col.min(),lon_col.max()
    img_lat_min,img_lat_max = lat_row.min(),lat_row.max()

    step = max(1, min(rows, cols) // 400)

    geo_polys = []
    for bbox_m, pts_m in polygons:
        bx0,by0 = _merc_to_geo(bbox_m[0], bbox_m[1])
        bx1,by1 = _merc_to_geo(bbox_m[2], bbox_m[3])
        if bx1 < img_lon_min or bx0 > img_lon_max: continue
        if by1 < img_lat_min or by0 > img_lat_max: continue
        pts_geo = [_merc_to_geo(x,y) for x,y in pts_m]
        geo_polys.append(((float(bx0),float(by0),float(bx1),float(by1)), pts_geo))

    for (bx0,by0,bx1,by1), pts_geo in geo_polys:
        for ri in range(0, rows, step):
            lat = float(lat_row[ri] if hasattr(lat_row,'__len__') else lat_row)
            if lat < by0 or lat > by1: continue
            for ci in range(0, cols, step):
                lon = float(lon_col[ci] if hasattr(lon_col,'__len__') else lon_col)
                if lon < bx0 or lon > bx1: continue
                if _point_in_polygon(lon, lat, pts_geo):
                    mask[ri:min(ri+step,rows), ci:min(ci+step,cols)] = True

    return binary_dilation(mask, iterations=max(1, step))


# ─────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    if WEIGHTS_PATH.exists():
        return YOLO(str(WEIGHTS_PATH))
    return None


def download_weights():
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    r=requests.get(WEIGHTS_URL,stream=True,timeout=120); r.raise_for_status()
    total=int(r.headers.get("content-length",0)); bar=st.progress(0); done=0
    with open(WEIGHTS_PATH,"wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk); done+=len(chunk)
            if total: bar.progress(min(done/total,1.0))


# ─────────────────────────────────────────────────────────────
#  SAR I/O & CALIBRATION
# ─────────────────────────────────────────────────────────────

def _dn_to_db(dn):
    """DN → γ0 dB using Sentinel-1 default constant factor (1e-4)."""
    valid = dn > 0
    sigma = np.where(valid, dn.astype(np.float64)**2 * 1e-4, 1e-10)
    db    = (10.0*np.log10(np.clip(sigma,1e-10,None))).astype(np.float32)
    return db, valid


def calibrate(dn, safe_dir):
    """Full γ0 calibration from SAFE XML LUT; falls back to _dn_to_db."""
    import rasterio.transform as rt
    cal  = list(Path(safe_dir).rglob("calibration-s1*-vv-*.xml"))
    valid = dn > 0
    if not cal:
        return _dn_to_db(dn)
    lut=[]
    for vec in ET.parse(cal[0]).getroot().iter("calibrationVector"):
        pe,ge=vec.find("pixel"),vec.find("gammaNought")
        if pe is not None and ge is not None:
            lut.append(([int(x) for x in pe.text.split()],
                        [float(x) for x in ge.text.split()]))
    h,w=dn.shape; la=np.ones((h,w),dtype=np.float64)
    for i,(px,v) in enumerate(lut):
        if i>=h: break
        la[i]=np.interp(np.arange(w),px,v)
    dn64=dn.astype(np.float64)
    g0=np.where(valid, dn64**2/(la**2+1e-30), 1e-10)
    db=(10.0*np.log10(np.clip(g0,1e-10,None))).astype(np.float32)
    return db, valid


def read_sar(uploaded):
    """Returns (image_db, valid_mask, tmp_path, transform, crs_epsg)."""
    import rasterio, rasterio.transform as rt
    suffix=Path(uploaded.name).suffix.lower()
    tmp=TEMP_DIR/uploaded.name; tmp.write_bytes(uploaded.read())
    if suffix in [".tiff",".tif"]:
        with rasterio.open(tmp) as src:
            dn=src.read(1).astype(np.float32)
            transform=src.transform
            crs_epsg=src.crs.to_epsg() if src.crs else 4326
        db,valid=_dn_to_db(dn)
        return db,valid,str(tmp),transform,crs_epsg
    if suffix==".png":
        from PIL import Image
        arr=np.array(Image.open(tmp).convert("L")).astype(np.float32)
        transform=rt.from_bounds(0,0,arr.shape[1],arr.shape[0],arr.shape[1],arr.shape[0])
        return arr,(arr>0),str(tmp),transform,4326
    if suffix==".zip":
        ext=TEMP_DIR/"safe"
        if ext.exists(): shutil.rmtree(ext)
        with zipfile.ZipFile(tmp) as z: z.extractall(ext)
        vv=list(ext.rglob("*-vv-*.tiff"))+list(ext.rglob("*-vv-*.tif"))
        if not vv: raise FileNotFoundError("No VV TIFF found in ZIP.")
        with rasterio.open(vv[0]) as src:
            dn=src.read(1).astype(np.float32)
            transform=src.transform
            crs_epsg=src.crs.to_epsg() if src.crs else 4326
        db,valid=calibrate(dn,vv[0].parent.parent)
        return db,valid,str(vv[0]),transform,crs_epsg
    raise ValueError(f"Unsupported: {suffix}")


# ─────────────────────────────────────────────────────────────
#  SPECKLE FILTER
# ─────────────────────────────────────────────────────────────

def lee(img, valid_mask, size=7):
    """Lee speckle filter; nodata pixels excluded from statistics."""
    from scipy.ndimage import uniform_filter
    out=img.copy()
    m=uniform_filter(img,size); sq=uniform_filter(img**2,size); v=sq-m**2
    noise_var=np.mean(v[valid_mask]) if valid_mask.any() else np.mean(v)
    w=v/(v+noise_var+1e-10)
    out=m+w*(img-m); out[~valid_mask]=img[~valid_mask]
    return out


# ─────────────────────────────────────────────────────────────
#  CA-CFAR DETECTION
# ─────────────────────────────────────────────────────────────

def _box_mean(img, half):
    """O(1) box-filter mean via cumulative sum."""
    padded=np.pad(img,half,mode="edge")
    cs=padded.cumsum(axis=0).cumsum(axis=1)
    size=2*half+1
    return (cs[size:,size:]-cs[:-size,size:]-cs[size:,:-size]+cs[:-size,:-size])/size**2


def cfar_detect(img, valid_mask, land_mask, guard, bg, thresh, mn, mx):
    """
    CA-CFAR with annular guard ring.
    Edge strips (burst boundaries) removed via valid_mask erosion.
    Land pixels excluded. Detections filtered by blob area [mn, mx].
    """
    from scipy.ndimage import label as scipy_label, binary_erosion

    # Erode valid mask to strip bright burst-boundary edge artefacts
    edge_margin = max(12, bg*2+guard)
    proc_mask   = binary_erosion(valid_mask, iterations=edge_margin, border_value=0)
    proc_mask  &= ~land_mask

    # Replace nodata so filters don't smear borders
    work=img.copy()
    bg_fill=np.median(img[valid_mask]) if valid_mask.any() else 0.0
    work[~valid_mask]=bg_fill

    # Annular background: outer box minus guard box
    mean_out  = _box_mean(work, bg)
    mean_grd  = _box_mean(work, guard)
    n_out     = (2*bg+1)**2; n_grd=(2*guard+1)**2; n_ring=n_out-n_grd
    mean_ring = (mean_out*n_out - mean_grd*n_grd) / (n_ring+1e-10)

    var_out   = np.clip(_box_mean(work**2, bg) - mean_out**2, 0, None)
    var_grd   = np.clip(_box_mean(work**2, guard) - mean_grd**2, 0, None)
    var_ring  = np.clip((var_out*n_out - var_grd*n_grd)/(n_ring+1e-10), 0, None)
    std_ring  = np.sqrt(var_ring)

    detected  = (work > mean_ring + thresh*std_ring) & proc_mask
    labeled,n = scipy_label(detected)
    boxes=[]
    for i in range(1,n+1):
        r,c=np.where(labeled==i); sz=len(r)
        if sz<mn or sz>mx: continue
        boxes.append(dict(x_min=int(c.min()),y_min=int(r.min()),
                          x_max=int(c.max()),y_max=int(r.max()),
                          w=int(c.max()-c.min()+1),h=int(r.max()-r.min()+1)))
    return boxes


# ─────────────────────────────────────────────────────────────
#  YOLO DETECTION
# ─────────────────────────────────────────────────────────────

def yolo_detect(img_path, conf, tile, overlap, land_mask):
    if not WEIGHTS_PATH.exists(): return []
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    dm=AutoDetectionModel.from_pretrained(
        model_type="ultralytics",model_path=str(WEIGHTS_PATH),
        confidence_threshold=conf,device="cpu")
    res=get_sliced_prediction(img_path,dm,
        slice_height=tile,slice_width=tile,
        overlap_height_ratio=overlap,overlap_width_ratio=overlap,verbose=0)
    boxes=[]
    h,w=land_mask.shape
    for d in res.object_prediction_list:
        x0,y0=int(d.bbox.minx),int(d.bbox.miny)
        x1,y1=int(d.bbox.maxx),int(d.bbox.maxy)
        cx,cy=(x0+x1)//2,(y0+y1)//2
        if 0<=cy<h and 0<=cx<w and land_mask[cy,cx]: continue
        boxes.append(dict(x_min=x0,y_min=y0,x_max=x1,y_max=y1,
                          w=x1-x0,h=y1-y0,conf=float(d.score.value)))
    return boxes


# ─────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────

def disp(img, valid_mask=None):
    src=img[valid_mask] if (valid_mask is not None and valid_mask.any()) else img
    p2,p98=np.percentile(src,[2,98])
    return ((np.clip(img,p2,p98)-p2)/(p98-p2+1e-10)*255).astype(np.uint8)


def figure(img, valid_mask, land_mask, cb, yb):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as pat
    import matplotlib.gridspec as gs
    from matplotlib.lines import Line2D

    d   = disp(img, valid_mask)
    fig = plt.figure(figsize=(22,8),facecolor="#070d1a")
    g   = gs.GridSpec(1,3,figure=fig,wspace=.025,left=.005,right=.995,top=.91,bottom=.08)
    cc,yc,lc="#00d4ff","#ff3d5a","#f0a500"

    land_rgba=np.zeros((*land_mask.shape,4),dtype=np.float32)
    if land_mask.any():
        land_rgba[land_mask,0]=0.94; land_rgba[land_mask,1]=0.65
        land_rgba[land_mask,2]=0.0;  land_rgba[land_mask,3]=0.30

    for col,(title,c,y) in enumerate([
        (f"CFAR  ·  {len(cb)} vessels", cb, []),
        (f"YOLOv8  ·  {len(yb)} wakes", [], yb),
        ("COMBINED OVERLAY",             cb, yb),
    ]):
        ax=fig.add_subplot(g[col])
        ax.imshow(d,cmap="gray",aspect="auto",interpolation="bilinear")
        if land_mask.any():
            ax.imshow(land_rgba,aspect="auto",interpolation="none")
        ax.set_title(title,color="#4a6080",fontsize=8.5,fontfamily="monospace",pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#0d1e35")
        for b in c:
            ax.add_patch(pat.Rectangle((b["x_min"],b["y_min"]),b["w"],b["h"],
                                       lw=1.5,edgecolor=cc,facecolor="none",alpha=.85))
        for b in y:
            ax.add_patch(pat.Rectangle((b["x_min"],b["y_min"]),b["w"],b["h"],
                                       lw=1.5,edgecolor=yc,facecolor="none",alpha=.85))
            if "conf" in b:
                ax.text(b["x_min"],b["y_min"]-2,f"{b['conf']:.2f}",
                        color=yc,fontsize=5.5,fontfamily="monospace")

    fig.legend(handles=[
        Line2D([0],[0],color=cc,lw=2,label=f"CFAR — {len(cb)} detections"),
        Line2D([0],[0],color=yc,lw=2,label=f"YOLOv8 — {len(yb)} detections"),
        pat.Patch(facecolor=lc,alpha=0.5,label="Land mask"),
    ],loc="lower center",ncol=3,facecolor="#070d1a",edgecolor="#0d1e35",
      labelcolor="#b8c4d8",fontsize=9,prop={"family":"monospace"},bbox_to_anchor=(.5,.005))
    fig.suptitle("SARDETECT  ·  DUAL METHOD VESSEL DETECTION",
                 color="#00d4ff",fontsize=12,fontfamily="monospace",y=.975)
    buf=BytesIO()
    plt.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown('<p style="font-family:monospace;color:#f0a500;font-size:1.1rem;letter-spacing:.1em">SARDETECT</p>',unsafe_allow_html=True)
        st.markdown('<p style="font-family:monospace;color:#3d4f6a;font-size:.7rem;letter-spacing:.2em">DUAL METHOD DETECTION</p>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**CFAR PARAMETERS**")
        guard  = st.slider("Guard window",        1,   8,    3)
        bg     = st.slider("Background window",   6,  25,   12)
        thresh = st.slider("Threshold factor",  1.0, 15.0,  5.0, 0.5)
        mn     = st.slider("Min size (px)",       2,  50,    8)
        mx     = st.slider("Max size (px)",     100, 5000, 1500, 100)
        st.markdown("---")
        st.markdown("**YOLO PARAMETERS**")
        conf    = st.slider("Confidence",  0.10, 0.90, 0.25, 0.05)
        tile    = st.selectbox("Tile size",[512,1024,2048],index=1)
        overlap = st.slider("Tile overlap",0.10, 0.40, 0.20, 0.05)
        st.markdown("---")
        land_on = st.checkbox("Apply land mask", value=True)
        st.markdown("---")
        if not WEIGHTS_PATH.exists():
            st.warning("Model weights not found.")
            if st.button("DOWNLOAD WEIGHTS"):
                with st.spinner("Downloading..."):
                    try:
                        download_weights(); st.success("Done."); st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.success("✓ Model weights loaded")
        land_status="✓ LandPolygon.zip found" if LAND_ZIP.exists() else "⚠ LandPolygon.zip missing"
        st.markdown(f'<p style="font-family:monospace;font-size:.7rem;color:#3d4f6a">{land_status}</p>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p style="font-family:monospace;font-size:.7rem;color:#3d4f6a">Model: YOLOv8m-OBB<br>Dataset: OpenSARWake<br>Xu & Wang, IEEE GRSL 2024<br>github.com/Yelow47/GE7090</p>',unsafe_allow_html=True)
    return guard, bg, thresh, mn, mx, conf, tile, overlap, land_on


# ─────────────────────────────────────────────────────────────
#  FAQ
# ─────────────────────────────────────────────────────────────

def faq_section():
    st.markdown("---")
    st.markdown("## FAQ")
    for q,a in [
        ("What file formats are supported?",
         "GeoTIFF (.tif/.tiff), PNG, and zipped Sentinel-1 SAFE folders (.zip). For SAFE folders the VV band is automatically extracted and radiometrically calibrated to gamma-naught (γ0) using the XML LUT."),
        ("What is CFAR detection?",
         "Constant False Alarm Rate (CFAR) detects vessels by finding pixels significantly brighter than their local surroundings. This implementation uses CA-CFAR with an annular guard ring: background mean and standard deviation are computed from the ring of cells between the guard and background windows. A pixel is flagged if it exceeds mean_background + T × std_background. Edge strips and land pixels are excluded before thresholding."),
        ("What is YOLOv8 wake detection?",
         "YOLOv8-OBB is a deep learning detector trained on the OpenSARWake dataset (Xu & Wang, 2024, IEEE GRSL). It detects the hydrodynamic wake a vessel leaves in the water. Wakes are visible even when the vessel itself is too small or defocused. Large scene inference is handled via SAHI sliced prediction."),
        ("Why do the two methods find different vessels?",
         "CFAR detects the ship directly via radar cross-section. YOLOv8 detects the ship indirectly via its wake. A stationary vessel may have no wake. A fast-moving vessel may be defocused in SAR but have a clear wake. Where both methods agree, confidence is highest."),
        ("How does the land mask work?",
         "The bundled LandPolygon.zip contains a global polygon shapefile in Web Mercator (EPSG:3857). At runtime, polygons overlapping the SAR scene are rasterised onto the image grid. CFAR excludes land pixels entirely. YOLO detections whose centre falls on land are suppressed. Land areas are shown as a semi-transparent amber overlay on all output panels."),
        ("Why was CFAR giving 277 false alarms at the image edge?",
         "Sentinel-1 IW mode produces bright burst-boundary artefacts at the top and sides of scenes. The updated CFAR erodes the valid-pixel mask by the background window radius before detection, stripping these edge strips. The CA-CFAR formula now uses an annular ring background estimate with standard-deviation scaling, which is more robust to heterogeneous clutter than the previous mean-only threshold."),
        ("What is radiometric calibration?",
         "Raw Sentinel-1 GRD data contains uncalibrated digital numbers. Calibration converts these to gamma-naught (γ0) via γ0 = DN² / LUT², where LUT is interpolated from the calibration XML in the SAFE package. For GeoTIFF without an XML the Sentinel-1 default constant factor (10⁻⁴) is applied, giving physically meaningful dB values rather than raw DN²."),
        ("What satellite data works best?",
         "Sentinel-1 IW mode GRD in VV polarization. Free data from Copernicus Data Space (dataspace.copernicus.eu). Recommended shipping lanes: Strait of Gibraltar, English Channel, North Sea."),
        ("What does mAP50 mean?",
         "Mean Average Precision at IoU 0.50. A detection counts as correct if the predicted box overlaps the ground truth by at least 50%. The model achieves competitive mAP50 compared to the paper\'s SWNet (49.0% mAP50)."),
    ]:
        st.markdown(f'<p class="faq-q">▸ {q}</p>',unsafe_allow_html=True)
        st.markdown(f'<p class="faq-a">{a}</p>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="font-family:monospace;font-size:.75rem;color:#3d4f6a;text-align:center">OpenSARWake — Xu & Wang (2024) · IEEE GRSL · DOI: 10.1109/LGRS.2024.3392681<br>Built with Streamlit · Ultralytics YOLOv8 · SAHI · Rasterio · SciPy</p>',unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    guard,bg,thresh,mn,mx,conf,tile,overlap,land_on = sidebar()

    st.markdown('<p class="main-title">SARDETECT</p>',unsafe_allow_html=True)
    st.markdown('<p class="sub-title">SYNTHETIC APERTURE RADAR  ·  DUAL METHOD VESSEL DETECTION</p>',unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<div class="card cfar-top"><h4>METHOD 1 — CFAR</h4><p>Physics-based CA-CFAR with annular guard ring. Edge strips and land pixels excluded before thresholding.</p></div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card yolo-top"><h4>METHOD 2 — YOLOv8</h4><p>Deep learning wake detection trained on 3,973 multi-band SAR images from the OpenSARWake dataset (Xu & Wang, 2024).</p></div>',unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card comb-top"><h4>COMBINED OUTPUT</h4><p>Side-by-side comparison with detections from both methods overlaid on the same SAR scene. Land mask shown in amber.</p></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### UPLOAD SAR IMAGE")
    st.markdown('<p class="upload-hint">Supported: GeoTIFF (.tif/.tiff) · PNG · Zipped Sentinel-1 SAFE folder (.zip)</p>',unsafe_allow_html=True)

    uploaded=st.file_uploader("",type=["tif","tiff","png","zip"],label_visibility="collapsed")

    if uploaded:
        st.markdown(f'<p style="font-family:monospace;font-size:.78rem;color:#8a9ab8">✓ {uploaded.name} — {uploaded.size/1e6:.1f} MB</p>',unsafe_allow_html=True)
        if st.button("▶  RUN DETECTION"):
            prog=st.progress(0); status=st.empty()
            try:
                status.markdown("`Reading SAR image…`")
                image,valid_mask,tmp_path,transform,crs_epsg=read_sar(uploaded)
                prog.progress(15)

                status.markdown("`Building land mask…`")
                if land_on:
                    polygons =_load_land_polygons()
                    land_mask=build_land_mask(transform,image.shape,polygons,crs_epsg)
                else:
                    land_mask=np.zeros(image.shape,dtype=bool)
                prog.progress(30)

                status.markdown("`Applying Lee speckle filter…`")
                filtered=lee(image,valid_mask)
                prog.progress(45)

                status.markdown("`Running CA-CFAR detection…`")
                cfar_boxes=cfar_detect(filtered,valid_mask,land_mask,guard,bg,thresh,mn,mx)
                prog.progress(62)

                from PIL import Image as PI
                png_tmp=str(TEMP_DIR/"yolo_input.png")
                d=disp(filtered,valid_mask)
                PI.fromarray(np.stack([d,d,d],axis=-1)).save(png_tmp)

                status.markdown("`Running YOLOv8 wake detection…`")
                yolo_boxes=yolo_detect(png_tmp,conf,tile,overlap,land_mask)
                prog.progress(85)

                status.markdown("`Generating figure…`")
                fig_buf=figure(filtered,valid_mask,land_mask,cfar_boxes,yolo_boxes)
                prog.progress(100); status.empty()

                st.markdown("### RESULTS")
                m1,m2,m3=st.columns(3)
                with m1:
                    st.markdown(f'<div class="mbox"><p class="mval cc">{len(cfar_boxes)}</p><p class="mlbl">CFAR Detections</p></div>',unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="mbox"><p class="mval yc">{len(yolo_boxes)}</p><p class="mlbl">YOLOv8 Detections</p></div>',unsafe_allow_html=True)
                with m3:
                    h,w=image.shape
                    land_pct=100*land_mask.sum()/land_mask.size if land_on else 0
                    st.markdown(f'<div class="mbox"><p class="mval dc" style="font-size:1.4rem">{w}×{h}</p><p class="mlbl">{land_pct:.1f}% masked as land</p></div>',unsafe_allow_html=True)

                st.markdown("")
                st.image(fig_buf,use_container_width=True)
                st.download_button("⬇  DOWNLOAD RESULT",data=fig_buf.getvalue(),
                                   file_name=f"sardetect_{Path(uploaded.name).stem}.png",
                                   mime="image/png")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback; st.code(traceback.format_exc())

    faq_section()


if __name__=="__main__":
    main()

