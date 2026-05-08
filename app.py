import streamlit as st
import numpy as np
import tempfile
import zipfile
import shutil
import os
import requests
from pathlib import Path
from io import BytesIO
import xml.etree.ElementTree as ET

st.set_page_config(page_title="SARDetect", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Barlow',sans-serif;background:#070d1a;color:#b8c4d8}
.stApp{background:radial-gradient(ellipse at 15% 10%,#0a1a35 0%,#070d1a 55%)}
h1,h2,h3{font-family:'Share Tech Mono',monospace!important}
.main-title{font-family:'Share Tech Mono',monospace;font-size:2.6rem;color:#00d4ff;letter-spacing:.1em;text-shadow:0 0 40px rgba(0,212,255,.35);margin:0}
.sub-title{font-family:'Share Tech Mono',monospace;font-size:.78rem;color:#2a4060;letter-spacing:.22em;margin:0 0 1.5rem}
.card{background:#0a1220;border:1px solid #0d1e35;border-top:2px solid;border-radius:3px;padding:14px 18px}
.card h4{font-family:'Share Tech Mono',monospace;font-size:.82rem;letter-spacing:.15em;margin:0 0 6px}
.card p{font-size:.83rem;color:#5a7090;margin:0;line-height:1.55}
.cfar-top{border-top-color:#00d4ff}.cfar-top h4{color:#00d4ff}
.yolo-top{border-top-color:#ff3d5a}.yolo-top h4{color:#ff3d5a}
.comb-top{border-top-color:#7fff6a}.comb-top h4{color:#7fff6a}
.mbox{background:#080f1e;border:1px solid #0d1e35;border-radius:3px;padding:18px;text-align:center}
.mval{font-family:'Share Tech Mono',monospace;font-size:2.4rem;margin:0;line-height:1}
.mlbl{font-size:.7rem;color:#3a5070;letter-spacing:.18em;margin:4px 0 0;text-transform:uppercase}
.cc{color:#00d4ff}.yc{color:#ff3d5a}.dc{color:#4a6080}
.faq-q{font-family:'Share Tech Mono',monospace;color:#00d4ff;font-size:.88rem;margin:18px 0 4px}
.faq-a{color:#6a8098;font-size:.86rem;line-height:1.65;margin:0}
.stButton>button{font-family:'Share Tech Mono',monospace!important;background:#00d4ff!important;color:#070d1a!important;border:none!important;border-radius:2px!important;font-weight:700!important;letter-spacing:.12em!important}
.stButton>button:hover{background:#00aacc!important;box-shadow:0 0 24px rgba(0,212,255,.28)!important}
section[data-testid="stSidebar"]{background:#080f1e;border-right:1px solid #0d1a2e}
.stProgress>div>div{background:#00d4ff!important}
hr{border-color:#0d1a2e!important}
</style>""", unsafe_allow_html=True)

WEIGHTS_URL  = "https://github.com/Yelow47/GE7090/releases/download/SAR/best.pt"
WEIGHTS_PATH = Path("weights/best.pt")
TEMP_DIR     = Path(tempfile.gettempdir()) / "sardetect"
TEMP_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    if WEIGHTS_PATH.exists():
        return YOLO(str(WEIGHTS_PATH))
    return None


def download_weights():
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    r = requests.get(WEIGHTS_URL, stream=True, timeout=120)
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


def read_sar(uploaded):
    import rasterio
    suffix = Path(uploaded.name).suffix.lower()
    tmp    = TEMP_DIR / uploaded.name
    tmp.write_bytes(uploaded.read())

    if suffix in [".tiff", ".tif"]:
        with rasterio.open(tmp) as src:
            data = src.read(1).astype(np.float32)
        return data, str(tmp)

    if suffix == ".png":
        from PIL import Image
        return np.array(Image.open(tmp).convert("L")).astype(np.float32), str(tmp)

    if suffix == ".zip":
        ext = TEMP_DIR / "safe"
        if ext.exists():
            shutil.rmtree(ext)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(ext)
        vv = list(ext.rglob("*-vv-*.tiff")) + list(ext.rglob("*-vv-*.tif"))
        if not vv:
            raise FileNotFoundError("No VV TIFF found in ZIP.")
        with rasterio.open(vv[0]) as src:
            data = src.read(1).astype(np.float32)
        return calibrate(data, vv[0].parent.parent), str(vv[0])

    raise ValueError(f"Unsupported: {suffix}")


def calibrate(dn, safe_dir):
    cal = list(Path(safe_dir).rglob("calibration-s1*-vv-*.xml"))
    if not cal:
        raw = np.where(dn > 0, dn.astype(np.float64)**2, 1e-10)
        return (10 * np.log10(raw)).astype(np.float32)
    lut = []
    for vec in ET.parse(cal[0]).getroot().iter("calibrationVector"):
        pe, ge = vec.find("pixel"), vec.find("gammaNought")
        if pe is not None and ge is not None:
            lut.append(([int(x) for x in pe.text.split()],
                        [float(x) for x in ge.text.split()]))
    h, w  = dn.shape
    la    = np.ones((h, w), dtype=np.float32)
    for i, (px, v) in enumerate(lut):
        if i >= h: break
        la[i] = np.interp(np.arange(w), px, v)
    g0 = np.where(dn > 0, dn.astype(np.float64)**2 / la.astype(np.float64)**2, 1e-10)
    return (10 * np.log10(np.clip(g0, 1e-10, None))).astype(np.float32)


def lee(img, size=7):
    from scipy.ndimage import uniform_filter
    m  = uniform_filter(img, size)
    v  = uniform_filter(img**2, size) - m**2
    w  = v / (v + np.mean(v) + 1e-10)
    return m + w * (img - m)


def cfar_detect(img, guard, bg, thresh, mn, mx):
    from scipy.ndimage import uniform_filter, label as lbl
    bm  = uniform_filter(img, bg*2+1)
    gm  = uniform_filter(img, guard*2+1)
    cl  = np.abs(bm - gm * ((guard*2+1)**2 / (bg*2+1)**2))
    det = img > (bm + cl * thresh)
    labeled, n = lbl(det)
    boxes = []
    for i in range(1, n+1):
        r  = np.where(labeled == i)
        sz = len(r[0])
        if sz < mn or sz > mx: continue
        y0,y1 = r[0].min(),r[0].max()
        x0,x1 = r[1].min(),r[1].max()
        boxes.append(dict(x_min=int(x0),y_min=int(y0),
                          x_max=int(x1),y_max=int(y1),
                          w=x1-x0+1,h=y1-y0+1))
    return boxes


def yolo_detect(img_path, conf, tile, overlap):
    if not WEIGHTS_PATH.exists(): return []
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    dm  = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(WEIGHTS_PATH),
        confidence_threshold=conf, device="cpu")
    res = get_sliced_prediction(img_path, dm,
                                slice_height=tile, slice_width=tile,
                                overlap_height_ratio=overlap,
                                overlap_width_ratio=overlap, verbose=0)
    return [dict(x_min=int(d.bbox.minx), y_min=int(d.bbox.miny),
                 x_max=int(d.bbox.maxx), y_max=int(d.bbox.maxy),
                 w=d.bbox.maxx-d.bbox.minx, h=d.bbox.maxy-d.bbox.miny,
                 conf=float(d.score.value))
            for d in res.object_prediction_list]


def disp(img):
    p2,p98 = np.percentile(img,[2,98])
    return ((np.clip(img,p2,p98)-p2)/(p98-p2+1e-10)*255).astype(np.uint8)


def figure(img, cb, yb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as pat
    import matplotlib.gridspec as gs
    from matplotlib.lines import Line2D

    d   = disp(img)
    fig = plt.figure(figsize=(22,8), facecolor="#070d1a")
    g   = gs.GridSpec(1,3,figure=fig,wspace=.025,
                      left=.005,right=.995,top=.91,bottom=.08)
    cc,yc = "#00d4ff","#ff3d5a"

    for col,(title,c,y) in enumerate([
        (f"CFAR  ·  {len(cb)} vessels", cb, []),
        (f"YOLOv8  ·  {len(yb)} wakes",  [], yb),
        ("COMBINED OVERLAY",              cb, yb),
    ]):
        ax = fig.add_subplot(g[col])
        ax.imshow(d, cmap="gray", aspect="auto", interpolation="bilinear")
        ax.set_title(title, color="#4a6080", fontsize=8.5,
                     fontfamily="monospace", pad=5)
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
    ], loc="lower center", ncol=2, facecolor="#070d1a", edgecolor="#0d1e35",
       labelcolor="#b8c4d8", fontsize=9, prop={"family":"monospace"},
       bbox_to_anchor=(.5,.005))
    fig.suptitle("SARDETECT  ·  DUAL METHOD VESSEL DETECTION",
                 color="#00d4ff", fontsize=12, fontfamily="monospace", y=.975)

    buf = BytesIO()
    plt.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def sidebar():
    with st.sidebar:
        st.markdown('<p style="font-family:monospace;color:#00d4ff;font-size:1.1rem;letter-spacing:.1em">SARDETECT</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:monospace;color:#1a3050;font-size:.7rem;letter-spacing:.2em">DUAL METHOD DETECTION</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**CFAR PARAMETERS**")
        guard  = st.slider("Guard window",       1,  8,   3)
        bg     = st.slider("Background window",  5, 20,  10)
        thresh = st.slider("Threshold factor", 1.0,15.0, 6.0, 0.5)
        mn     = st.slider("Min size (px)",      2, 50,  10)
        mx     = st.slider("Max size (px)",    100,5000,2000,100)
        st.markdown("---")
        st.markdown("**YOLO PARAMETERS**")
        conf    = st.slider("Confidence",  0.1, 0.9, 0.25, 0.05)
        tile    = st.selectbox("Tile size",[512,1024,2048],index=1)
        overlap = st.slider("Tile overlap",0.1, 0.4, 0.2,  0.05)
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
        st.markdown("---")
        st.markdown('<p style="font-family:monospace;font-size:.7rem;color:#1a3050">Model: YOLOv8m-OBB<br>Dataset: OpenSARWake<br>Xu & Wang, IEEE GRSL 2024<br>github.com/Yelow47/GE7090</p>', unsafe_allow_html=True)
    return guard, bg, thresh, mn, mx, conf, tile, overlap


def faq_section():
    st.markdown("---")
    st.markdown("## FAQ")
    for q, a in [
        ("What file formats are supported?",
         "GeoTIFF (.tif/.tiff), PNG, and zipped Sentinel-1 SAFE folders (.zip). For SAFE folders the VV band is automatically extracted and radiometrically calibrated to gamma-naught (γ0)."),
        ("What is CFAR detection?",
         "Constant False Alarm Rate (CFAR) detects vessels by finding pixels significantly brighter than their local surroundings. Ships appear bright in SAR due to strong radar reflection from metal hulls. CFAR adapts its threshold locally to maintain consistent false alarm rates across varying sea conditions. This replicates the ArcGIS Detect Bright Ocean Objects workflow."),
        ("What is YOLOv8 wake detection?",
         "YOLOv8-OBB is a deep learning detector trained on the OpenSARWake dataset (Xu & Wang, 2024, IEEE GRSL). Instead of detecting the vessel directly, it detects the hydrodynamic wake a vessel leaves in the water. Wakes are often visible even when the vessel is too small or moving too fast to appear as a clear bright return."),
        ("Why do the two methods find different vessels?",
         "CFAR detects the ship directly via radar cross-section. YOLOv8 detects the ship indirectly via its wake. A stationary vessel may have no wake. A fast-moving vessel may be defocused in SAR but have a clear wake. Where both methods agree, confidence is highest."),
        ("What satellite data works best?",
         "Sentinel-1 IW mode GRD in VV polarization. Free data is available from Copernicus Data Space (dataspace.copernicus.eu). Busy shipping lanes such as the Strait of Gibraltar, English Channel, or North Sea are recommended."),
        ("What is radiometric calibration?",
         "Raw Sentinel-1 data contains uncalibrated digital numbers. Calibration converts these to gamma-naught (γ0) backscatter using a lookup table from the SAFE metadata XML. Gamma-naught normalises for incidence angle variation across the swath, which is essential for reliable CFAR thresholding over open ocean."),
        ("How do I tune detection sensitivity?",
         "Use the sidebar. For CFAR: lower Threshold Factor to detect more vessels at the cost of more false alarms. Adjust Min/Max size to filter by physical vessel size. For YOLOv8: lower Confidence to detect more wakes."),
        ("What does mAP50 mean?",
         "Mean Average Precision at IoU 0.50. A detection counts as correct if the predicted box overlaps the ground truth by at least 50%. The model trained here achieves competitive mAP50 compared to the paper's purpose-built SWNet (49.0% mAP50)."),
    ]:
        st.markdown(f'<p class="faq-q">▸ {q}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="faq-a">{a}</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="font-family:monospace;font-size:.75rem;color:#1a3050;text-align:center">OpenSARWake — Xu & Wang (2024) · IEEE GRSL · DOI: 10.1109/LGRS.2024.3392681<br>Built with Streamlit · Ultralytics YOLOv8 · SAHI · Rasterio · SciPy</p>', unsafe_allow_html=True)


def main():
    guard, bg, thresh, mn, mx, conf, tile, overlap = sidebar()

    st.markdown('<p class="main-title">SARDETECT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">SYNTHETIC APERTURE RADAR  ·  DUAL METHOD VESSEL DETECTION</p>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card cfar-top"><h4>METHOD 1 — CFAR</h4><p>Physics-based detection via elevated radar backscatter from vessel hulls. Replicates the ArcGIS Detect Bright Ocean Objects workflow.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card yolo-top"><h4>METHOD 2 — YOLOv8</h4><p>Deep learning wake detection trained on 3,973 multi-band SAR images from the OpenSARWake dataset (Xu & Wang, 2024).</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card comb-top"><h4>COMBINED OUTPUT</h4><p>Side-by-side comparison with detections from both methods overlaid on the same SAR scene.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### UPLOAD SAR IMAGE")
    st.markdown('<p style="font-family:monospace;font-size:.78rem;color:#1a3050">Supported: GeoTIFF (.tif/.tiff) · PNG · Zipped Sentinel-1 SAFE folder (.zip)</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["tif","tiff","png","zip"], label_visibility="collapsed")

    if uploaded:
        st.markdown(f'<p style="font-family:monospace;font-size:.78rem;color:#2a5070">✓ {uploaded.name} — {uploaded.size/1e6:.1f} MB</p>', unsafe_allow_html=True)

        if st.button("▶  RUN DETECTION"):
            prog   = st.progress(0)
            status = st.empty()
            try:
                status.markdown("`Reading SAR image...`")
                image, tmp_path = read_sar(uploaded)
                prog.progress(20)

                status.markdown("`Applying Lee speckle filter...`")
                filtered = lee(image)
                prog.progress(35)

                status.markdown("`Running CFAR detection...`")
                cfar_boxes = cfar_detect(filtered, guard, bg, thresh, mn, mx)
                prog.progress(55)

                from PIL import Image as PI
                png_tmp = str(TEMP_DIR / "yolo_input.png")
                d = disp(filtered)
                PI.fromarray(np.stack([d,d,d],axis=-1)).save(png_tmp)

                status.markdown("`Running YOLOv8 wake detection...`")
                yolo_boxes = yolo_detect(png_tmp, conf, tile, overlap)
                prog.progress(80)

                status.markdown("`Generating figure...`")
                fig_buf = figure(filtered, cfar_boxes, yolo_boxes)
                prog.progress(100)
                status.empty()

                st.markdown("### RESULTS")
                m1,m2,m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="mbox"><p class="mval cc">{len(cfar_boxes)}</p><p class="mlbl">CFAR Detections</p></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="mbox"><p class="mval yc">{len(yolo_boxes)}</p><p class="mlbl">YOLOv8 Detections</p></div>', unsafe_allow_html=True)
                with m3:
                    h,w = image.shape
                    st.markdown(f'<div class="mbox"><p class="mval dc" style="font-size:1.4rem">{w}×{h}</p><p class="mlbl">Image Dimensions</p></div>', unsafe_allow_html=True)

                st.markdown("")
                st.image(fig_buf, use_container_width=True)
                st.download_button("⬇  DOWNLOAD RESULT",
                                   data=fig_buf.getvalue(),
                                   file_name=f"sardetect_{Path(uploaded.name).stem}.png",
                                   mime="image/png")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    faq_section()


if __name__ == "__main__":
    main()
