import streamlit as st
import numpy as np
import tempfile
import zipfile
import shutil
import requests
import traceback
from pathlib import Path
from io import BytesIO
import xml.etree.ElementTree as ET

APP_VERSION = "v1.16"

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
.cc{color:#00c8f0}.yc{color:#ff4a5a}.ac{color:#4dff91}.dc{color:#8a9ab8}
.faq-q{font-family:'Share Tech Mono',monospace;color:#f0a500;font-size:.88rem;margin:20px 0 6px;}
.faq-a{color:#8a9ab8;font-size:.88rem;line-height:1.7;margin:0;}
.stButton>button{font-family:'Share Tech Mono',monospace!important;background:#f0a500!important;color:#080e1f!important;border:none!important;border-radius:3px!important;font-weight:700!important;letter-spacing:.1em!important;padding:.55rem 2rem!important;transition:all .2s!important;}
.stButton>button:hover{background:#d49000!important;box-shadow:0 0 24px rgba(240,165,0,0.3)!important;}
div[data-testid="stFileUploader"]{background:#0a1122!important;border:1px dashed #1a2e50!important;border-radius:4px!important;}
section[data-testid="stSidebar"]{background:#0c1428;border-right:1px solid #111e35;}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stMarkdown{color:#e8edf5!important;}
.stSlider>div>div>div{background:#f0a500!important}.stProgress>div>div{background:#f0a500!important}
.stAlert{background:#0c1428!important;border-color:#1a2e50!important;color:#e8edf5!important}
hr{border-color:#111e35!important;margin:1.5rem 0!important}
.upload-hint{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#3d4f6a;letter-spacing:.1em;margin:4px 0 12px;}
.log-box{background:#0a1122;border:1px solid #111e35;border-radius:4px;padding:12px 16px;font-family:'Share Tech Mono',monospace;font-size:.78rem;color:#8a9ab8;max-height:220px;overflow-y:auto;margin-top:8px;white-space:pre-wrap;}
.log-error{color:#ff4a5a !important;}
.log-warn{color:#f0a500 !important;}
.log-ok{color:#4dff91 !important;}
</style>""", unsafe_allow_html=True)

WEIGHTS_URL  = "https://github.com/Yelow47/GE7090/releases/download/SAR/best.pt"
WEIGHTS_PATH = Path("weights/best.pt")
LAND_ZIP     = Path("LandPolygon.zip")
TILE_SIZE    = 1024
TEMP_DIR     = Path(tempfile.gettempdir()) / "sardetect"
TEMP_DIR.mkdir(exist_ok=True)


# ── LAND MASK ─────────────────────────────────────────────────

def build_land_mask(transform, shape, crs_epsg):
    """
    Rasterise global land polygon onto the SAR image grid.
    Uses the least-squares affine transform built from the full geolocation
    grid, so rasterize receives the same coordinate system as the image.
    No flipping required — confirmed correct by backscatter test.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds
    from shapely.geometry import box as shapely_box

    rows, cols = shape
    if not LAND_ZIP.exists():
        return np.zeros((rows, cols), dtype=bool)

    land_dir = TEMP_DIR / "land"
    land_dir.mkdir(exist_ok=True)
    shp_path = land_dir / "LandPolygon.shp"
    if not shp_path.exists():
        with zipfile.ZipFile(LAND_ZIP) as z:
            z.extractall(land_dir)

    gdf = gpd.read_file(str(shp_path))
    epsg = int(crs_epsg) if crs_epsg else 4326
    image_crs = CRS.from_epsg(epsg)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3857")
    if gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)

    bounds  = array_bounds(rows, cols, transform)
    img_box = shapely_box(*bounds)
    gdf     = gdf[gdf.intersects(img_box)]

    if gdf.empty:
        return np.zeros((rows, cols), dtype=bool)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    mask   = rasterize(shapes, out_shape=(rows, cols),
                       transform=transform, fill=0, dtype=np.uint8)
    return mask.astype(bool)


# ── MODEL ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    if WEIGHTS_PATH.exists():
        return YOLO(str(WEIGHTS_PATH))
    return None


def download_weights():
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    r     = requests.get(WEIGHTS_URL, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    bar   = st.progress(0)
    done  = 0
    with open(WEIGHTS_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            done += len(chunk)
            if total:
                bar.progress(min(done / total, 1.0))


# ── SAR I/O & CALIBRATION ─────────────────────────────────────

def _dn_to_db(dn):
    valid = dn > 0
    flat  = dn[valid] if valid.any() else dn.ravel()
    if flat.min() >= -60.0 and flat.max() <= 60.0:
        return dn.astype(np.float32), valid
    sigma = np.where(valid, dn.astype(np.float64)**2 * 1e-4, 1e-10)
    db    = (10.0 * np.log10(np.clip(sigma, 1e-10, None))).astype(np.float32)
    return db, valid


def _match_shapes(db, valid):
    h = min(db.shape[0], valid.shape[0])
    w = min(db.shape[1], valid.shape[1])
    return db[:h, :w], valid[:h, :w]


def calibrate(dn, safe_dir):
    cal   = list(Path(safe_dir).rglob("calibration-s1*-vv-*.xml"))
    valid = dn > 0
    if not cal:
        return _dn_to_db(dn)
    lut = []
    for vec in ET.parse(cal[0]).getroot().iter("calibrationVector"):
        pe, ge = vec.find("pixel"), vec.find("gammaNought")
        if pe is not None and ge is not None:
            lut.append(([int(x) for x in pe.text.split()],
                        [float(x) for x in ge.text.split()]))
    h, w = dn.shape
    la   = np.ones((h, w), dtype=np.float64)
    for i, (px, v) in enumerate(lut):
        if i >= h:
            break
        la[i] = np.interp(np.arange(w), px, v)
    dn64 = dn.astype(np.float64)
    g0   = np.where(valid, dn64**2 / (la**2 + 1e-30), 1e-10)
    db   = (10.0 * np.log10(np.clip(g0, 1e-10, None))).astype(np.float32)
    return db, valid


def read_sar(uploaded):
    import rasterio
    suffix = Path(uploaded.name).suffix.lower()
    tmp    = TEMP_DIR / uploaded.name
    tmp.write_bytes(uploaded.read())

    if suffix in [".tiff", ".tif"]:
        with rasterio.open(tmp) as src:
            dn        = src.read(1).astype(np.float32)
            transform = src.transform
            crs_epsg  = src.crs.to_epsg() if src.crs else 4326
        db, valid = _dn_to_db(dn)
        db, valid = _match_shapes(db, valid)
        return db, valid, transform, crs_epsg

    if suffix == ".zip":
        ext = TEMP_DIR / "safe"
        if ext.exists():
            shutil.rmtree(ext)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(ext)
        vv = list(ext.rglob("*-vv-*.tiff")) + list(ext.rglob("*-vv-*.tif"))
        if not vv:
            raise FileNotFoundError(
                "No VV polarization TIFF found in ZIP. "
                "Upload a Sentinel-1 SAFE folder zipped directly.")
        with rasterio.open(vv[0]) as src:
            dn            = src.read(1).astype(np.float32)
            crs_epsg      = src.crs.to_epsg() if src.crs else 4326
            raw_transform = src.transform

        from rasterio.transform import Affine
        safe_dir  = vv[0].parent.parent
        ann_files = list((safe_dir / "annotation").glob("*-vv-*.xml"))
        if ann_files:
            lons, lats, rows_gcp, cols_gcp = [], [], [], []
            for pt in ET.parse(ann_files[0]).getroot().iter("geolocationGridPoint"):
                lons.append(float(pt.find("longitude").text))
                lats.append(float(pt.find("latitude").text))
                rows_gcp.append(float(pt.find("line").text))
                cols_gcp.append(float(pt.find("pixel").text))
            rows_gcp = np.array(rows_gcp)
            cols_gcp = np.array(cols_gcp)
            lons     = np.array(lons)
            lats     = np.array(lats)
            # Least-squares affine fit over all ~200 geolocation grid points.
            # More accurate than from_bounds which only uses 4 corner values
            # and ignores the parallelogram distortion of Sentinel-1 IW scenes.
            A = np.column_stack([cols_gcp, rows_gcp, np.ones(len(cols_gcp))])
            lon_c, _, _, _ = np.linalg.lstsq(A, lons, rcond=None)
            lat_c, _, _, _ = np.linalg.lstsq(A, lats, rcond=None)
            transform = Affine(lon_c[0], lon_c[1], lon_c[2],
                               lat_c[0], lat_c[1], lat_c[2])
        else:
            transform = raw_transform

        db, valid = calibrate(dn, safe_dir)
        db, valid = _match_shapes(db, valid)
        return db, valid, transform, crs_epsg

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        "Upload a GeoTIFF (.tif/.tiff) or zipped Sentinel-1 SAFE folder (.zip).")


# ── SPECKLE FILTER — for YOLO only ────────────────────────────

def lee(img, valid_mask, size=7):
    from scipy.ndimage import uniform_filter
    out       = img.copy()
    m         = uniform_filter(img, size)
    sq        = uniform_filter(img**2, size)
    v         = sq - m**2
    noise_var = np.mean(v[valid_mask]) if valid_mask.any() else np.mean(v)
    w         = v / (v + noise_var + 1e-10)
    out       = m + w * (img - m)
    out[~valid_mask] = img[~valid_mask]
    return out


# ── CA-CFAR ────────────────────────────────────────────────────

def _box_mean(img, half):
    from scipy.ndimage import uniform_filter
    return uniform_filter(img.astype(np.float32), size=2*half+1, mode='nearest')


def cfar_detect(img, valid_mask, land_mask, thresh, min_w, max_w, min_l, max_l):
    # Guard and background windows are internal CFAR parameters — hardcoded.
    # Sentinel-1 IW GRD pixel spacing is 10m.
    GUARD         = 4
    BG            = 16
    PIXEL_SPACING = 10  # metres per pixel
    """
    CA-CFAR detection with two speed optimisations applied in sequence:

    1. Crop to the bounding box of valid (non-nodata) pixels.
       Nodata triangles in SAR corner fill areas are skipped entirely.
       Coordinates are mapped back to the full image before returning.

    2. Downsample the cropped region to half resolution before running
       the four expensive uniform_filter passes (~4× faster).
       Detected boxes are scaled back to full-resolution coordinates.
       At Sentinel-1 IW 10 m GSD a ship occupies ~20 px; at half resolution
       it still covers ~10 px, so no meaningful detection loss occurs.
    """
    from scipy.ndimage import label as scipy_label
    from scipy.ndimage import zoom

    h, w       = img.shape
    valid_mask = valid_mask[:h, :w]
    land_mask  = land_mask[:h, :w]

    # ── Optimisation 1: crop to bounding box of valid pixels ──────────────
    rows_valid = np.where(valid_mask.any(axis=1))[0]
    cols_valid = np.where(valid_mask.any(axis=0))[0]

    if rows_valid.size == 0 or cols_valid.size == 0:
        # Entire image is nodata — nothing to detect
        return []

    r0, r1 = int(rows_valid[0]),  int(rows_valid[-1])  + 1
    c0, c1 = int(cols_valid[0]),  int(cols_valid[-1])  + 1

    img_crop   = img[r0:r1, c0:c1]
    valid_crop = valid_mask[r0:r1, c0:c1]
    land_crop  = land_mask[r0:r1, c0:c1]

    # ── Optimisation 2: downsample to half resolution ─────────────────────
    SCALE = 0.5
    img_s   = zoom(img_crop,                         SCALE, order=1)
    valid_s = zoom(valid_crop.astype(np.float32),    SCALE, order=1) > 0.5
    land_s  = zoom(land_crop.astype(np.float32),     SCALE, order=1) > 0.5

    # Scale CFAR window parameters proportionally
    guard_s = max(1, int(round(GUARD * SCALE)))
    bg_s    = max(3, int(round(BG    * SCALE)))

    # ── Core CFAR on downsampled image ────────────────────────────────────
    hs, ws    = img_s.shape
    edge      = max(12, bg_s * 2 + guard_s)
    proc_mask = valid_s.copy()
    proc_mask[:edge, :]   = False
    proc_mask[-edge:, :]  = False
    proc_mask[:, :edge]   = False
    proc_mask[:, -edge:]  = False
    proc_mask &= ~land_s

    bg_fill = float(np.median(img_s[valid_s])) if valid_s.any() else 0.0
    work    = img_s.astype(np.float32)
    work[~valid_s] = bg_fill

    work_sq   = work**2
    mean_out  = _box_mean(work, bg_s)
    mean_grd  = _box_mean(work, guard_s)
    n_out     = (2 * bg_s    + 1)**2
    n_grd     = (2 * guard_s + 1)**2
    n_ring    = n_out - n_grd
    mean_ring = (mean_out * n_out - mean_grd * n_grd) / (n_ring + 1e-10)

    var_out  = np.clip(_box_mean(work_sq, bg_s)    - mean_out**2, 0, None)
    var_grd  = np.clip(_box_mean(work_sq, guard_s) - mean_grd**2, 0, None)
    var_ring = np.clip((var_out * n_out - var_grd * n_grd) / (n_ring + 1e-10), 0, None)
    std_ring = np.sqrt(var_ring)
    del work_sq, var_out, var_grd

    detected        = (work > mean_ring + thresh * std_ring) & proc_mask
    labeled, n_feat = scipy_label(detected)

    # ── Scale boxes back to full image coordinates ────────────────────────
    INV = 1.0 / SCALE
    boxes = []
    for i in range(1, n_feat + 1):
        r, c = np.where(labeled == i)
        # Scale blob dimensions back to full resolution in metres
        blob_w_m = (c.max() - c.min() + 1) * INV * PIXEL_SPACING
        blob_l_m = (r.max() - r.min() + 1) * INV * PIXEL_SPACING
        if blob_w_m < min_w or blob_w_m > max_w:
            continue
        if blob_l_m < min_l or blob_l_m > max_l:
            continue
        # Scale from downsampled crop space → full image space
        x_min = int(c.min() * INV) + c0
        y_min = int(r.min() * INV) + r0
        x_max = int(c.max() * INV) + c0
        y_max = int(r.max() * INV) + r0
        boxes.append(dict(
            x_min=x_min, y_min=y_min,
            x_max=x_max, y_max=y_max,
            w=x_max - x_min + 1,
            h=y_max - y_min + 1,
        ))
    return boxes


# ── AGREEMENT DETECTION ───────────────────────────────────────

def find_agreements(cfar_boxes, yolo_boxes, iou_thresh=0.1):
    """Find CFAR boxes that overlap with any YOLO box — these are agreement detections."""
    agreements = set()
    for ci, cb in enumerate(cfar_boxes):
        for yb in yolo_boxes:
            ix0 = max(cb["x_min"], yb["x_min"])
            iy0 = max(cb["y_min"], yb["y_min"])
            ix1 = min(cb["x_max"], yb["x_max"])
            iy1 = min(cb["y_max"], yb["y_max"])
            if ix1 > ix0 and iy1 > iy0:
                agreements.add(ci)
                break
    return agreements


# ── YOLO ───────────────────────────────────────────────────────

def _prepare_yolo_image(img, valid_mask):
    from PIL import Image as PILImage
    src       = img[valid_mask] if valid_mask.any() else img.ravel()
    p1, p99   = np.percentile(src, [1, 99])
    stretched = np.clip(img, p1, p99)
    stretched = ((stretched - p1) / (p99 - p1 + 1e-10) * 255).astype(np.uint8)
    rgb       = np.stack([stretched, stretched, stretched], axis=-1)
    png_path  = str(TEMP_DIR / "yolo_input.png")
    PILImage.fromarray(rgb).save(png_path)
    return png_path


def yolo_detect(img, valid_mask, conf, overlap, land_mask, log):
    if not WEIGHTS_PATH.exists():
        log("WARN: YOLOv8 weights not found — skipping wake detection.")
        return []
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as e:
        log(f"ERROR: SAHI import failed: {e}")
        return []
    try:
        log("  Loading YOLOv8 weights...")
        dm = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(WEIGHTS_PATH),
            confidence_threshold=conf,
            device="cpu")
        log("  Model loaded.")
    except Exception as e:
        log(f"ERROR loading model: {e}")
        return []
    try:
        png_path = _prepare_yolo_image(img, valid_mask)
        log(f"  Running SAHI sliced inference (tile={TILE_SIZE}px, overlap={overlap})...")
        res   = get_sliced_prediction(
            png_path, dm,
            slice_height=TILE_SIZE, slice_width=TILE_SIZE,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            verbose=1)
        n_raw = len(res.object_prediction_list)
        log(f"  SAHI raw detections: {n_raw}")
    except Exception as e:
        log(f"ERROR during SAHI inference: {e}")
        log(traceback.format_exc())
        return []

    boxes  = []
    h, w   = land_mask.shape
    n_land = 0
    for d in res.object_prediction_list:
        x0, y0 = int(d.bbox.minx), int(d.bbox.miny)
        x1, y1 = int(d.bbox.maxx), int(d.bbox.maxy)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if 0 <= cy < h and 0 <= cx < w and land_mask[cy, cx]:
            n_land += 1
            continue
        boxes.append(dict(x_min=x0, y_min=y0, x_max=x1, y_max=y1,
                          w=x1 - x0, h=y1 - y0, conf=float(d.score.value)))
    if n_land > 0:
        log(f"  Suppressed {n_land} detections over land.")
    return boxes


# ── EXPORT SHAPEFILES ────────────────────────────────────────

def export_shapefiles_zip(cfar_boxes, yolo_boxes, transform, agreements, stem):
    """
    Export detections as a ZIP containing three shapefiles (point features):
      - <stem>_cfar.shp       — CFAR detections
      - <stem>_yolo.shp       — YOLOv8 detections
      - <stem>_combined.shp   — all detections combined
    All in EPSG:4326. Ready to drag-and-drop into ArcGIS Pro.
    """
    import io, os, tempfile
    import geopandas as gpd
    from shapely.geometry import Point

    def px_to_geo(col, row, tf):
        lon = tf.c + col * tf.a + row * tf.b
        lat = tf.f + col * tf.d + row * tf.e
        return float(lon), float(lat)

    cfar_rows, yolo_rows, combined_rows = [], [], []

    for i, b in enumerate(cfar_boxes):
        cx = (b["x_min"] + b["x_max"]) / 2
        cy = (b["y_min"] + b["y_max"]) / 2
        lon, lat = px_to_geo(cx, cy, transform)
        method = "CFAR+YOLO" if i in agreements else "CFAR"
        row = {"geometry": Point(lon, lat), "method": method,
               "width_m": b["w"] * 10, "height_m": b["h"] * 10,
               "x_min": b["x_min"], "y_min": b["y_min"],
               "x_max": b["x_max"], "y_max": b["y_max"]}
        cfar_rows.append(row)
        combined_rows.append(row)

    for b in yolo_boxes:
        cx = (b["x_min"] + b["x_max"]) / 2
        cy = (b["y_min"] + b["y_max"]) / 2
        lon, lat = px_to_geo(cx, cy, transform)
        row = {"geometry": Point(lon, lat), "method": "YOLOv8",
               "conf": round(b.get("conf", 0), 3),
               "width_m": b["w"] * 10, "height_m": b["h"] * 10,
               "x_min": b["x_min"], "y_min": b["y_min"],
               "x_max": b["x_max"], "y_max": b["y_max"]}
        yolo_rows.append(row)
        combined_rows.append(row)

    tmp = Path(tempfile.mkdtemp())
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, rows in [
            (f"{stem}_cfar",     cfar_rows),
            (f"{stem}_yolo",     yolo_rows),
            (f"{stem}_combined", combined_rows),
        ]:
            if not rows:
                continue
            gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
            out_dir = tmp / name
            out_dir.mkdir()
            shp_path = out_dir / f"{name}.shp"
            gdf.to_file(str(shp_path), driver="ESRI Shapefile")
            for f in out_dir.iterdir():
                zf.write(f, arcname=f"{name}/{f.name}")

    buf.seek(0)
    return buf.getvalue()


# ── DISPLAY & FIGURE ──────────────────────────────────────────

def disp(img, valid_mask=None):
    src     = img[valid_mask] if (valid_mask is not None and valid_mask.any()) else img
    p2, p98 = np.percentile(src, [2, 98])
    return ((np.clip(img, p2, p98) - p2) / (p98 - p2 + 1e-10) * 255).astype(np.uint8)


def figure(img, valid_mask, land_mask, cb, yb, agreements):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as pat
    import matplotlib.gridspec as gs
    from matplotlib.lines import Line2D

    d   = disp(img, valid_mask)
    fig = plt.figure(figsize=(22, 8), facecolor="#070d1a")
    g   = gs.GridSpec(1, 3, figure=fig, wspace=.025,
                      left=.005, right=.995, top=.91, bottom=.08)
    cc, yc, ac, lc = "#00d4ff", "#ff3d5a", "#4dff91", "#f0a500"

    land_rgba = np.zeros((*land_mask.shape, 4), dtype=np.float32)
    if land_mask.any():
        land_rgba[land_mask, 0] = 0.94
        land_rgba[land_mask, 1] = 0.65
        land_rgba[land_mask, 3] = 0.30

    for col, (title, c, y) in enumerate([
        (f"CFAR  ·  {len(cb)} vessels", cb, []),
        (f"YOLOv8  ·  {len(yb)} wakes", [], yb),
        ("COMBINED OVERLAY",             cb, yb),
    ]):
        ax = fig.add_subplot(g[col])
        ax.imshow(d, cmap="gray", aspect="auto", interpolation="bilinear")
        if land_mask.any():
            ax.imshow(land_rgba, aspect="auto", interpolation="none")
        ax.set_title(title, color="#4a6080", fontsize=8.5,
                     fontfamily="monospace", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#0d1e35")

        for i, b in enumerate(c):
            color = ac if (col == 2 and i in agreements) else cc
            ax.add_patch(pat.Rectangle(
                (b["x_min"], b["y_min"]), b["w"], b["h"],
                lw=1.5, edgecolor=color, facecolor="none", alpha=.85))

        for b in y:
            ax.add_patch(pat.Rectangle(
                (b["x_min"], b["y_min"]), b["w"], b["h"],
                lw=1.5, edgecolor=yc, facecolor="none", alpha=.85))
            if "conf" in b:
                ax.text(b["x_min"], b["y_min"] - 2, f"{b['conf']:.2f}",
                        color=yc, fontsize=5.5, fontfamily="monospace")

    n_agree = len(agreements)
    fig.legend(
        handles=[
            Line2D([0], [0], color=cc, lw=2, label=f"CFAR — {len(cb)} detections"),
            Line2D([0], [0], color=yc, lw=2, label=f"YOLOv8 — {len(yb)} detections"),
            Line2D([0], [0], color=ac, lw=2, label=f"Agreement — {n_agree} detections"),
            pat.Patch(facecolor=lc, alpha=0.5, label="Land mask"),
        ],
        loc="lower center", ncol=4,
        facecolor="#070d1a", edgecolor="#0d1e35",
        labelcolor="#b8c4d8", fontsize=9,
        prop={"family": "monospace"},
        bbox_to_anchor=(.5, .005))
    fig.suptitle("SARDETECT  ·  DUAL METHOD VESSEL DETECTION",
                 color="#00d4ff", fontsize=12, fontfamily="monospace", y=.975)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ── SIDEBAR ───────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown('<p style="font-family:monospace;color:#f0a500;font-size:1.1rem;letter-spacing:.1em">SARDETECT</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:monospace;color:#3d4f6a;font-size:.7rem;letter-spacing:.2em">DUAL METHOD DETECTION</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**CFAR PARAMETERS**")
        thresh  = st.slider("Threshold factor",       1.0, 15.0,   6.0, 0.5)
        min_w   = st.slider("Min object width (m)",     1,  100,    5)
        max_w   = st.slider("Max object width (m)",    50, 1500,  500)
        min_l   = st.slider("Min object length (m)",    1,  100,   10)
        max_l   = st.slider("Max object length (m)",   50, 2000,  700)
        st.markdown("---")
        st.markdown("**YOLO PARAMETERS**")
        conf    = st.slider("Confidence",   0.10, 0.90, 0.25, 0.05)
        overlap = st.slider("Tile overlap", 0.10, 0.40, 0.20, 0.05)
        st.markdown("---")
        land_on = st.checkbox("Apply land mask", value=True)
        st.markdown("---")
        if not WEIGHTS_PATH.exists():
            st.warning("Model weights not found.")
            if st.button("DOWNLOAD WEIGHTS"):
                with st.spinner("Downloading..."):
                    try:
                        download_weights()
                        st.success("Done.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.success("✓ Model weights loaded")
        land_status = "✓ LandPolygon.zip found" if LAND_ZIP.exists() else "⚠ LandPolygon.zip missing"
        st.markdown(f'<p style="font-family:monospace;font-size:.7rem;color:#3d4f6a">{land_status}</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p style="font-family:monospace;font-size:.7rem;color:#3d4f6a">Model: YOLOv8m-OBB<br>Dataset: OpenSARWake<br>Xu & Wang, IEEE GRSL 2024<br>github.com/Yelow47/GE7090</p>', unsafe_allow_html=True)
    return thresh, min_w, max_w, min_l, max_l, conf, overlap, land_on


# ── FAQ ───────────────────────────────────────────────────────

def faq_section():
    st.markdown("---")
    st.markdown("## FAQ")
    for q, a in [
        ("What file formats are supported?",
         "GeoTIFF (.tif/.tiff) and zipped Sentinel-1 SAFE folders (.zip). "
         "For SAFE folders the VV polarization band is automatically extracted and "
         "radiometrically calibrated to gamma-naught (γ0) using the XML LUT from the SAFE metadata."),
        ("What is CFAR detection?",
         "Constant False Alarm Rate (CFAR) detects vessels by finding pixels significantly brighter "
         "than their local surroundings. This uses CA-CFAR with an annular guard ring. "
         "CFAR runs directly on the raw calibrated γ0 dB image with no speckle filtering, consistent with "
         "the ArcGIS Detect Bright Ocean Objects workflow."),
        ("What is YOLOv8 wake detection?",
         "YOLOv8-OBB trained on OpenSARWake (Xu & Wang, 2024, IEEE GRSL). Detects the hydrodynamic wake "
         "a vessel leaves rather than the vessel itself. Large scene inference via SAHI using fixed "
         "1024×1024 tiles matching the OpenSARWake training chip size. The Lee speckle filter is applied "
         "before YOLO to improve wake pattern visibility."),
        ("Why does CFAR run on unfiltered data but YOLO on filtered data?",
         "CFAR detects sharp bright peaks relative to local background — speckle filtering smooths these "
         "peaks and degrades detection. YOLOv8 benefits from filtering because wake textures are clearer "
         "in speckle-reduced imagery."),
        ("Why do the two methods find different vessels?",
         "CFAR detects the ship directly via radar cross-section. YOLOv8 detects it indirectly via wake. "
         "A stationary vessel may have no wake. A fast-moving vessel may be defocused but have a clear wake. "
         "Where both agree, confidence is highest."),
        ("What does the green colour in the combined overlay mean?",
         "Green boxes indicate detections where both CFAR and YOLOv8 agree — the same location was flagged "
         "by both methods. These are the highest confidence vessel detections since two independent approaches "
         "based on different physical phenomena have identified the same target."),
        ("How does the land mask work?",
         "LandPolygon.zip contains a global polygon shapefile. It is reprojected to match the SAR image CRS "
         "and rasterised directly onto the image grid using geopandas and rasterio. CFAR excludes land pixels "
         "entirely. YOLO detections whose centre falls on land are suppressed. Land shown as amber overlay."),
        ("What is the GeoJSON export for?",
         "The GeoJSON file contains all detections as geographic point features with longitude/latitude "
         "coordinates. It can be opened directly in ArcGIS, QGIS, or any GIS software to overlay detections "
         "on top of the original SAR imagery or other geographic layers."),
        ("What is radiometric calibration?",
         "Raw Sentinel-1 GRD DN values are converted to gamma-naught γ0 = DN² / LUT², using the "
         "calibration XML in the SAFE package, then to dB scale: γ0_dB = 10 × log₁₀(γ0). "
         "Pre-calibrated GeoTIFFs are detected automatically and passed through unchanged."),
        ("What satellite data works best?",
         "Sentinel-1 IW mode GRD in VV polarization. Free from Copernicus Data Space "
         "(dataspace.copernicus.eu). Recommended: Strait of Gibraltar, English Channel, North Sea."),
        ("What does mAP50 mean?",
         "Mean Average Precision at IoU 0.50. The model achieves 58.4% mAP50 on the OpenSARWake test set, "
         "outperforming the paper's purpose-built SWNet architecture (49.0%)."),
    ]:
        st.markdown(f'<p class="faq-q">▸ {q}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="faq-a">{a}</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<p style="font-family:monospace;font-size:.75rem;color:#3d4f6a;text-align:center">'
        'OpenSARWake — Xu & Wang (2024) · IEEE GRSL · DOI: 10.1109/LGRS.2024.3392681<br>'
        'Built with Streamlit · Ultralytics YOLOv8 · SAHI · Rasterio · GeoPandas · SciPy'
        '</p>', unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────

def main():
    thresh, min_w, max_w, min_l, max_l, conf, overlap, land_on = sidebar()

    st.markdown(
        f'<div style="position:fixed;top:60px;right:18px;z-index:9999;">'
        f'<span style="font-family:Share Tech Mono,monospace;font-size:.7rem;'
        f'color:#3d4f6a;background:#0c1428;border:1px solid #111e35;'
        f'border-radius:3px;padding:3px 8px;letter-spacing:.1em">{APP_VERSION}</span>'
        f'</div>',
        unsafe_allow_html=True)
    st.markdown('<p class="main-title">SARDETECT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">SYNTHETIC APERTURE RADAR  ·  DUAL METHOD VESSEL DETECTION</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="card cfar-top"><h4>METHOD 1 — CFAR</h4>'
            '<p>Physics-based CA-CFAR on raw calibrated γ0 dB backscatter. '
            'Edge strips and land pixels excluded before thresholding.</p></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="card yolo-top"><h4>METHOD 2 — YOLOv8</h4>'
            '<p>Deep learning wake detection trained on 3,973 multi-band SAR images '
            'from OpenSARWake (Xu & Wang, 2024). Run on Lee-filtered image via SAHI.</p></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="card comb-top"><h4>COMBINED OUTPUT</h4>'
            '<p>Side-by-side comparison. Green boxes show agreement between both methods — '
            'highest confidence detections.</p></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### UPLOAD SAR IMAGE")
    st.markdown(
        '<p class="upload-hint">Supported: GeoTIFF (.tif/.tiff) · Zipped Sentinel-1 SAFE folder (.zip)</p>',
        unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["tif", "tiff", "zip"], label_visibility="collapsed")

    if uploaded:
        st.markdown(
            f'<p style="font-family:monospace;font-size:.78rem;color:#8a9ab8">'
            f'✓ {uploaded.name} — {uploaded.size/1e6:.1f} MB</p>',
            unsafe_allow_html=True)

        if st.button("▶  RUN DETECTION"):
            prog    = st.progress(0)
            status  = st.empty()
            log_box = st.empty()
            logs    = []

            def log(msg):
                logs.append(msg)
                lines = "".join(
                    f'<span class="'
                    f'{"log-error" if l.startswith("ERROR") else "log-warn" if l.startswith("WARN") else "log-ok" if l.startswith("✓") else ""}'
                    f'">{l}</span><br>'
                    for l in logs)
                log_box.markdown(f'<div class="log-box">{lines}</div>', unsafe_allow_html=True)

            try:
                status.markdown("`Reading SAR image…`")
                log("Reading SAR image...")
                image, valid_mask, transform, crs_epsg = read_sar(uploaded)
                vmin, vmax = image[valid_mask].min(), image[valid_mask].max()
                log(f"✓ Loaded: {image.shape[1]}×{image.shape[0]} px  |  γ0 range [{vmin:.1f}, {vmax:.1f}] dB  |  image={image.shape} valid={valid_mask.shape}")
                prog.progress(15)

                status.markdown("`Building land mask…`")
                log("Building land mask...")
                if land_on and LAND_ZIP.exists():
                    land_mask = build_land_mask(transform, image.shape, crs_epsg)
                    pct       = 100 * land_mask.sum() / land_mask.size
                    log(f"✓ Land mask built — {pct:.1f}% of image masked as land")
                else:
                    land_mask = np.zeros(image.shape, dtype=bool)
                    reason    = "disabled by user" if not land_on else "LandPolygon.zip not found"
                    log(f"WARN: Land masking skipped ({reason})")
                prog.progress(28)

                status.markdown("`Running CA-CFAR…`")
                log("Running CA-CFAR on raw calibrated γ0 dB image (cropped to valid area)...")
                cfar_boxes = cfar_detect(image, valid_mask, land_mask, thresh, min_w, max_w, min_l, max_l)
                log(f"✓ CFAR complete — {len(cfar_boxes)} detections")
                prog.progress(50)

                status.markdown("`Applying Lee speckle filter for YOLOv8…`")
                log("Applying Lee speckle filter (YOLO only)...")
                filtered = lee(image, valid_mask)
                log("✓ Lee filter applied")
                prog.progress(60)

                status.markdown("`Running YOLOv8 wake detection…`")
                log("Running YOLOv8 wake detection via SAHI...")
                yolo_boxes = yolo_detect(filtered, valid_mask, conf, overlap, land_mask, log)
                log(f"✓ YOLOv8 complete — {len(yolo_boxes)} detections")
                prog.progress(80)

                status.markdown("`Finding agreements…`")
                agreements = find_agreements(cfar_boxes, yolo_boxes)
                log(f"✓ Agreements found — {len(agreements)} detections confirmed by both methods")
                prog.progress(85)

                status.markdown("`Generating figure…`")
                log("Generating figure...")
                fig_buf = figure(image, valid_mask, land_mask, cfar_boxes, yolo_boxes, agreements)
                log("✓ Pipeline complete.")
                prog.progress(100)
                status.empty()

                st.markdown("### RESULTS")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f'<div class="mbox"><p class="mval cc">{len(cfar_boxes)}</p>'
                        f'<p class="mlbl">CFAR Detections</p></div>',
                        unsafe_allow_html=True)
                with m2:
                    st.markdown(
                        f'<div class="mbox"><p class="mval yc">{len(yolo_boxes)}</p>'
                        f'<p class="mlbl">YOLOv8 Detections</p></div>',
                        unsafe_allow_html=True)
                with m3:
                    st.markdown(
                        f'<div class="mbox"><p class="mval ac">{len(agreements)}</p>'
                        f'<p class="mlbl">Agreements</p></div>',
                        unsafe_allow_html=True)
                with m4:
                    h, w     = image.shape
                    land_pct = 100 * land_mask.sum() / land_mask.size if land_on else 0
                    st.markdown(
                        f'<div class="mbox"><p class="mval dc" style="font-size:1.4rem">'
                        f'{w}×{h}</p><p class="mlbl">{land_pct:.1f}% masked as land</p></div>',
                        unsafe_allow_html=True)

                st.markdown("")
                st.image(fig_buf, use_container_width=True)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "⬇  DOWNLOAD RESULT IMAGE",
                        data=fig_buf.getvalue(),
                        file_name=f"sardetect_{Path(uploaded.name).stem}.png",
                        mime="image/png")
                with col_dl2:
                    stem = Path(uploaded.name).stem
                    shp_zip = export_shapefiles_zip(
                        cfar_boxes, yolo_boxes, transform, agreements, stem)
                    st.download_button(
                        "⬇  DOWNLOAD SHAPEFILES (ArcGIS)",
                        data=shp_zip,
                        file_name=f"sardetect_{stem}_shapefiles.zip",
                        mime="application/zip")

            except Exception as e:
                log(f"ERROR: {e}")
                log(traceback.format_exc())
                status.error(f"Processing failed: {e}")

    faq_section()


if __name__ == "__main__":
    main()
