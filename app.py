import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import os
import glob

# ── Class names from metadata.yaml ──────────────────────────────────────────
CLASS_NAMES = {
    0: "jacklinks_beefjerkyorginal", 1: "mezzomix", 2: "jlbrichard_gauffre",
    3: "tarteaupomme", 4: "coke", 5: "vivitz_classiczitrone",
    6: "schweppes_citrus", 7: "bounty", 8: "volvic_pink", 9: "branche_maxi",
    10: "fuse_lemon", 11: "berger_nusstoertli", 12: "coke_light_flasche",
    13: "panettone", 14: "torino", 15: "tiki_himbeer_brause",
    16: "skittles_sour_riegel", 17: "ramseier_orange_tetrapak",
    18: "caferoyal_classicmacchiato", 19: "balisto_yoberry", 20: "fanta",
    21: "ramseier_schorle", 22: "fuse_lemon_dose", 23: "fini_jellykisses_packung",
    24: "zweifel_saltedpeanuts", 25: "mnms_gelb", 26: "c+swiss_dosenabisicetea",
    27: "berger_bruensli", 28: "oreo", 29: "kinder_delice",
    30: "maltesers", 31: "ovomaltine_kekse", 32: "coke_flasche",
    33: "ramseier_jusdepomme_tetrapak", 34: "kagi", 35: "vitaminwell_reload",
    36: "corny_schoko", 37: "valser_classic", 38: "stimorol_wildcherry_riegel",
    39: "malburner_partysticks", 40: "rivella_rot", 41: "haribo_goldbaerensauer",
    42: "comella_schokodrink", 43: "milka_tender", 44: "vivitz_gruentee",
    45: "fini_galaxymix_packung", 46: "knoppers_riegel", 47: "berger_schoggitoertli",
    48: "sinalco", 49: "darwida_chocaulait", 50: "erle",
    51: "henniez_gruen", 52: "ramseier_jusdepomme", 53: "wasser",
    54: "volvic_teeminze", 55: "airwaves_menthoneucalyptus_riegel",
    56: "fuse_peach_dose", 57: "milka_oreo_riegel", 58: "volvic_pinapple",
    59: "granini_orange", 60: "ragusa", 61: "jlbrichard_gauffrechoko",
    62: "zweifel_salz", 63: "maltesers_teasers", 64: "kitkat",
    65: "bifi_roll", 66: "bueno", 67: "evian", 68: "caprisun_multivitamin",
    69: "berger_vogelnestli", 70: "coke_dose", 71: "henniez_blau",
    72: "redbull", 73: "fuse_blackicetea", 74: "days_croissantschoko_packung",
    75: "valser_still", 76: "kagi_specialedition", 77: "lorenz_nicnacs",
    78: "snickers", 79: "powerbar_proteinplusschoko", 80: "toffifee",
    81: "stimorol_spearmint_riegel", 82: "darwida_sandwich",
    83: "milka_peanutbutter", 84: "comella_schokodrink_flasche",
    85: "snickers_wei", 86: "lorenz_studentenfutter",
    87: "zweifel_graneochilli", 88: "redbull_light", 89: "lindt_nocciolate",
    90: "twix", 91: "sinalco_orange", 92: "sprite_flasche",
    93: "schweppes_bitterlemon", 94: "berger_mailaenderli", 95: "coke_zero_flasche",
    96: "momo_icetea", 97: "valser_vivabirne", 98: "henniez_rot",
    99: "berger_spitzbueb", 100: "zweifel_paprika", 101: "pepsi_max",
    102: "fuse_peach", 103: "balisto_muesli",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.onnx")
BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "dataset", "images", "test")
DEFAULT_CONF = 0.15
DEFAULT_IOU = 0.60
INPUT_SIZE = 640

# ── Colour palette ───────────────────────────────────────────────────────────
np.random.seed(42)
COLORS = (np.random.randint(50, 255, (len(CLASS_NAMES), 3))).tolist()


@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(image: np.ndarray):
    """Letterbox-resize to INPUT_SIZE and normalise to [0,1]."""
    h, w = image.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    pad_h, pad_w = INPUT_SIZE - nh, INPUT_SIZE - nw
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    return blob, scale, left, top


def nms(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(outputs, scale, pad_left, pad_top, orig_h, orig_w, conf_thresh, iou_thresh):
    """Parse YOLOv8 output tensor [1, 4+num_classes, 8400] → detections."""
    pred = outputs[0][0]  # (4+C, 8400)
    boxes_raw = pred[:4].T   # (8400, 4) cx cy w h in padded space
    scores_all = pred[4:].T  # (8400, C)
    class_ids = scores_all.argmax(axis=1)
    confidences = scores_all[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thresh
    boxes_raw = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_raw) == 0:
        return [], [], []

    # cx,cy,w,h are normalised [0,1] → convert to pixels in padded 640×640 space
    cx = boxes_raw[:, 0] * INPUT_SIZE
    cy = boxes_raw[:, 1] * INPUT_SIZE
    bw = boxes_raw[:, 2] * INPUT_SIZE
    bh = boxes_raw[:, 3] * INPUT_SIZE

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Remove padding and scale back to original image
    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top) / scale, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, confidences, iou_thresh)
    return boxes[keep].astype(int), confidences[keep], class_ids[keep]


def draw_detections(image: np.ndarray, boxes, confidences, class_ids, border_mult=1.0):
    out = image.copy()
    h, w = out.shape[:2]
    scale_f = max(w, h) / 1000.0 * border_mult
    thickness = max(1, int(1.5 * scale_f))
    font_scale = max(0.4, 1.0 * scale_f)
    font_thickness = max(1, int(2 * scale_f))
    for box, conf, cid in zip(boxes, confidences, class_ids):
        color = COLORS[cid % len(COLORS)]
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{CLASS_NAMES.get(cid, cid)} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        pad = int(4 * scale_f)
        cv2.rectangle(out, (x1, max(0, y1 - th - bl - pad)), (x1 + tw + pad, y1), color, -1)
        cv2.putText(out, label, (x1 + pad // 2, y1 - bl - pad // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    return out


def show_result(pil_orig, result_img, n_det, detections, key_prefix):
    """Shared UI block: side-by-side images + zoom crop + detection list."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_orig, caption="Original", width=500)
    with col2:
        st.image(result_img, caption=f"Detections ({n_det})", width=500)

    if detections:
        st.subheader("Detected objects")
        for name, conf in sorted(detections, key=lambda x: -x[1]):
            st.markdown(f"- **{name}** — confidence {conf:.3f}")
    else:
        st.info("No objects detected above the confidence threshold.")


def run_inference(pil_image: Image.Image, session, conf_thresh=DEFAULT_CONF, iou_thresh=DEFAULT_IOU, border_mult=1.0):
    img_rgb = np.array(pil_image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    blob, scale, pl, pt = preprocess(img_rgb)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    boxes, confs, cids = postprocess(outputs, scale, pl, pt, h, w, conf_thresh, iou_thresh)
    result = draw_detections(img_rgb, boxes, confs, cids, border_mult)
    return Image.fromarray(result), len(boxes), list(zip(
        [CLASS_NAMES.get(int(c), str(c)) for c in cids],
        [float(f"{s:.3f}") for s in confs]
    ))


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="YOLO Object Detection", layout="wide", page_icon="🎯")

st.title("YOLOv8 Object Detection")

session = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model info")
    st.markdown("""
| | |
|---|---|
| **Model** | YOLOv8 (ONNX) |
| **Classes** | 104 |
| **Input size** | 640 × 640 |
""")
    st.markdown("---")
    st.subheader("Detection thresholds")
    conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.01,
                            help="Lower = more detections (may include false positives)")
    iou_thresh = st.slider("NMS IoU threshold", 0.10, 0.95, DEFAULT_IOU, 0.05,
                           help="Higher = keep more overlapping boxes (good for dense scenes)")
    border_mult = st.slider("Border thickness", 0.1, 5.0, 1.0, 0.1,
                            help="Multiplier on bounding box and label size")
    st.markdown("---")
    st.subheader("All detectable classes")
    for cid, name in sorted(CLASS_NAMES.items()):
        st.markdown(f"- {name}")

tab_camera, tab_upload, tab_builtin = st.tabs(["📷 Camera", "📂 Upload", "🖼️ Built-in Examples"])

# ── Camera tab ───────────────────────────────────────────────────────────────
with tab_camera:
    st.subheader("Take a photo")
    camera_img = st.camera_input("Point camera at an object and snap!")
    if camera_img:
        pil_img = Image.open(camera_img)
        with st.spinner("Running inference…"):
            result_img, n_det, detections = run_inference(pil_img, session, conf_thresh, iou_thresh, border_mult)
        show_result(pil_img, result_img, n_det, detections, "cam")

# ── Upload tab ───────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        pil_img = Image.open(uploaded)
        with st.spinner("Running inference…"):
            result_img, n_det, detections = run_inference(pil_img, session, conf_thresh, iou_thresh, border_mult)
        show_result(pil_img, result_img, n_det, detections, "up")

# ── Built-in examples tab ────────────────────────────────────────────────────
with tab_builtin:
    st.subheader("Built-in example images")
    image_paths = sorted(glob.glob(os.path.join(BUILTIN_DIR, "*.jpg")))
    if not image_paths:
        st.warning(f"No images found in {BUILTIN_DIR}")
    else:
        names = [os.path.basename(p) for p in image_paths]
        selected = st.selectbox("Choose an example image", names)
        chosen_path = os.path.join(BUILTIN_DIR, selected)
        pil_img = Image.open(chosen_path)
        with st.spinner("Running inference…"):
            result_img, n_det, detections = run_inference(pil_img, session, conf_thresh, iou_thresh, border_mult)
        show_result(pil_img, result_img, n_det, detections, "bi")
