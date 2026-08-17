"""
Waste Classification and Dustbin Guide
----------------------------------------
Streamlit application that uses a custom-trained YOLOv8 model (best.pt)
to identify the type of waste from an uploaded or captured photograph,
and informs the user which category of dustbin (as classified by the
Government of India under the Solid Waste Management Rules, 2016 and
Bio-Medical Waste Management Rules, 2016) the item should be disposed into.

Run with:
    streamlit run app.py

Place the trained weights file "best.pt" in the same folder as this script,
or update MODEL_PATH below. A .streamlit/config.toml is included alongside
this file to keep the interface on a consistent, readable light theme.
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except ImportError:
    st.error("The 'ultralytics' package is not installed. Run: pip install ultralytics")
    st.stop()


# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

MODEL_PATH = "best.pt"

# Bin classification followed in India:
#   Green Bin  -> Biodegradable / Wet waste
#   Blue Bin   -> Non-biodegradable / Dry, recyclable waste
#   Black Bin  -> Domestic hazardous / Reject waste (sanitary, soiled items, e-waste, etc.)
#   Yellow Bin -> Biomedical waste (as per Bio-Medical Waste Management Rules, 2016)

BIN_INFO = {
    "Green": {
        "color": "#2E7D32",
        "full_name": "Green Bin - Biodegradable (Wet) Waste",
        "description": "For organic, compostable, kitchen and garden waste.",
    },
    "Blue": {
        "color": "#1565C0",
        "full_name": "Blue Bin - Non-Biodegradable (Dry/Recyclable) Waste",
        "description": "For recyclable dry waste such as plastic, paper, metal and glass.",
    },
    "Black": {
        "color": "#212121",
        "full_name": "Black Bin - Domestic Hazardous / Reject Waste",
        "description": "For sanitary, soiled or hazardous domestic waste that cannot be recycled or composted. Wrap the item securely before disposal.",
    },
    "Yellow": {
        "color": "#F9A825",
        "full_name": "Yellow Bin - Biomedical Waste",
        "description": "For biomedical and clinical waste as per Bio-Medical Waste Management Rules, 2016. Must be handed over to authorised biomedical waste collectors.",
    },
}

# Mapping of every trained class to a bin category and a short disposal note.
WASTE_BIN_MAP = {
    "Brick":            {"bin": "Blue",   "note": "Construction and demolition waste. Store separately and hand over to municipal C&D waste collection where available."},
    "Broken Glass":     {"bin": "Blue",   "note": "Recyclable dry waste. Wrap in paper or cloth before disposal to avoid injury."},
    "Cardboard":        {"bin": "Blue",   "note": "Recyclable dry waste. Flatten before disposal."},
    "Cigarette":        {"bin": "Black",  "note": "Non-recyclable reject waste. Ensure it is fully extinguished before disposal."},
    "Cloth":            {"bin": "Blue",   "note": "Dry waste. Clean, usable cloth may instead be donated or given for textile recycling."},
    "Coconut Shell":    {"bin": "Green",  "note": "Biodegradable organic waste, suitable for composting."},
    "Dairy Packets":    {"bin": "Blue",   "note": "Rinse before disposal. Recyclable packaging waste."},
    "Footwear":         {"bin": "Black",  "note": "Reject waste made of mixed, non-recyclable materials."},
    "Glass Bottle":     {"bin": "Blue",   "note": "Recyclable dry waste."},
    "Mask":             {"bin": "Black",  "note": "Sanitary/domestic hazardous waste. Wrap separately in paper before disposal."},
    "Medical Waste":    {"bin": "Yellow", "note": "Biomedical waste. Must be handed over to an authorised biomedical waste handler, not household bins."},
    "Metal Can":        {"bin": "Blue",   "note": "Recyclable dry waste. Rinse before disposal."},
    "Packaging Box":    {"bin": "Blue",   "note": "Recyclable dry waste."},
    "Paper":            {"bin": "Blue",   "note": "Recyclable dry waste, if clean and not food-soiled."},
    "Paper Bag":        {"bin": "Blue",   "note": "Recyclable dry waste, if clean and not food-soiled."},
    "Paper Utensils":   {"bin": "Green",  "note": "Usually soiled with food waste; treat as wet/biodegradable waste."},
    "Plastic Bag":      {"bin": "Blue",   "note": "Dry waste. Reuse where possible; avoid single-use plastic."},
    "Plastic Bits":     {"bin": "Blue",   "note": "Dry, non-biodegradable waste."},
    "Plastic Bottle":   {"bin": "Blue",   "note": "Recyclable dry waste. Rinse and crush before disposal."},
    "Plastic Box":      {"bin": "Blue",   "note": "Recyclable dry waste."},
    "Plastic Cup":      {"bin": "Blue",   "note": "Recyclable dry waste, if not heavily food-soiled."},
    "Plastic Packet":   {"bin": "Blue",   "note": "Dry, non-biodegradable waste."},
    "Plastic Straw":    {"bin": "Blue",   "note": "Dry, non-biodegradable waste. Avoid single-use plastic where possible."},
    "Plastic Utensils": {"bin": "Blue",   "note": "Dry waste, if not heavily food-soiled."},
    "Sanitary":         {"bin": "Black",  "note": "Sanitary waste. Wrap securely in paper before disposal, as per Solid Waste Management Rules, 2016."},
}


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def contrast_text_color(rgb):
    r, g, b = [c / 255 for c in rgb]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1B1B1B" if luminance > 0.55 else "#FFFFFF"


def get_bin_details(class_name: str):
    entry = WASTE_BIN_MAP.get(class_name)
    if entry is None:
        return None
    bin_name = entry["bin"]
    bin_info = BIN_INFO[bin_name]
    return {
        "bin_name": bin_name,
        "bin_full_name": bin_info["full_name"],
        "bin_color": bin_info["color"],
        "note": entry["note"],
    }


# --------------------------------------------------------------------------------
# PAGE CONFIGURATION AND STYLE
# --------------------------------------------------------------------------------

st.set_page_config(
    page_title="Waste Classification and Dustbin Guide",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        color: #1B1B1B !important;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    section[data-testid="stSidebar"] {
        background-color: #EEF2F0;
    }
    section[data-testid="stSidebar"] * {
        color: #1B1B1B !important;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #1B1B1B;
    }
    .card {
        background-color: #FAFAF9;
        border: 1px solid #DDE3E0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .bin-card {
        padding: 18px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(0,0,0,0.08);
    }
    .bin-card h3 {
        margin: 0 0 6px 0;
    }
    .bin-card p {
        margin: 0;
        font-size: 0.95rem;
    }
    .result-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .input-help {
        background-color: #EEF2F0;
        border-left: 4px solid #2E7D32;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.92rem;
        margin-bottom: 10px;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #2E7D32;
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.2em;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #1B5E20;
        color: #FFFFFF !important;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #F5F7F6;
        border: 1.5px dashed #2E7D32;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------------

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)


model = load_model(MODEL_PATH)


# --------------------------------------------------------------------------------
# SESSION STATE FOR HISTORY
# --------------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, image, class, confidence, bin


# --------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------

st.sidebar.title("Waste Classification and Dustbin Guide")
st.sidebar.write(
    "Upload a photograph, or take one with your camera, to identify the type of "
    "waste and the correct dustbin to dispose of it into, based on the "
    "classification followed by the Government of India."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Bin Classification Reference")
for bin_name, info in BIN_INFO.items():
    text_color = contrast_text_color(hex_to_rgb(info["color"]))
    st.sidebar.markdown(
        f"""
        <div class="bin-card" style="background-color:{info['color']};">
            <h3 style="color:{text_color};">{bin_name} Bin</h3>
            <p style="color:{text_color};">{info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider(
    "Detection confidence threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05
)

if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.sidebar.success("History cleared.")


# --------------------------------------------------------------------------------
# MAIN AREA - INPUT
# --------------------------------------------------------------------------------

st.title("Waste Classification and Dustbin Guide")
st.write(
    "This tool uses a custom-trained YOLOv8 model to identify common types of "
    "waste and recommends the correct dustbin category as defined under the "
    "Solid Waste Management Rules, 2016 and the Bio-Medical Waste Management "
    "Rules, 2016."
)

if model is None:
    st.error(
        f"Model file '{MODEL_PATH}' was not found. Place the trained 'best.pt' "
        "file in the same directory as this script and reload the page."
    )
    st.stop()

tab_upload, tab_camera = st.tabs(["Upload Image", "Take Photograph"])

image_source = None

with tab_upload:
    st.markdown(
        '<div class="input-help">Use this tab to choose an existing photo of the waste item from your device '
        'gallery or files (JPG or PNG).</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Select an image file", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    st.markdown(
        '<div class="input-help">Use this tab to capture a single live photograph using your device camera. '
        'This takes a still photo only, not a video recording.</div>',
        unsafe_allow_html=True,
    )
    camera_file = st.camera_input("Capture a photograph", key="camera")
    if camera_file is not None:
        image_source = Image.open(camera_file).convert("RGB")


# --------------------------------------------------------------------------------
# CUSTOM ANNOTATION - BOXES/MASKS COLORED BY BIN CATEGORY
# --------------------------------------------------------------------------------

def annotate_image(pil_image, result, model, unmapped_color="#757575"):
    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    boxes = result.boxes
    masks = result.masks

    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])

        bin_details = get_bin_details(class_name)
        color_hex = bin_details["bin_color"] if bin_details else unmapped_color
        rgb = hex_to_rgb(color_hex)

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if masks is not None and idx < len(masks.xy):
            polygon = [tuple(point) for point in masks.xy[idx]]
            if len(polygon) >= 3:
                draw.polygon(polygon, outline=rgb + (255,), fill=rgb + (70,))
        else:
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=4)

        label = f"{class_name} {conf * 100:.0f}%"
        try:
            bbox = draw.textbbox((0, 0), label)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = len(label) * 7, 14

        label_y = max(y1 - text_h - 8, 0)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 8], fill=rgb + (235,))
        draw.text((x1 + 5, label_y + 3), label, fill=contrast_text_color(rgb))

    return img


# --------------------------------------------------------------------------------
# INFERENCE AND RESULTS
# --------------------------------------------------------------------------------

if image_source is not None:
    with st.spinner("Analysing image..."):
        results = model.predict(source=np.array(image_source), conf=confidence_threshold, verbose=False)

    result = results[0]
    boxes = result.boxes

    st.markdown("---")
    col_input, col_annotated, col_bin = st.columns([1, 1, 1])

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(image_source, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if boxes is None or len(boxes) == 0:
        with col_annotated:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Waste")
            st.image(image_source, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_bin:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Disposal Guidance")
            st.warning("No waste item could be confidently identified. Try a clearer photograph or lower the confidence threshold.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        annotated_img = annotate_image(image_source, result, model)

        with col_annotated:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Waste")
            st.image(annotated_img, use_container_width=True)
            st.caption("Box or outline color matches the recommended bin color.")
            st.markdown("</div>", unsafe_allow_html=True)

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"class": class_name, "confidence": conf})
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        top_detection = detections[0]
        bin_details = get_bin_details(top_detection["class"])

        with col_bin:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Disposal Guidance")
            st.markdown(f"<div class='result-header'>{top_detection['class']}</div>", unsafe_allow_html=True)
            st.write(f"Confidence: {top_detection['confidence'] * 100:.1f} percent")

            if bin_details is None:
                st.warning("No bin mapping is defined for this class yet.")
            else:
                text_color = contrast_text_color(hex_to_rgb(bin_details["bin_color"]))
                st.markdown(
                    f"""
                    <div class="bin-card" style="background-color:{bin_details['bin_color']};">
                        <h3 style="color:{text_color};">{bin_details['bin_full_name']}</h3>
                        <p style="color:{text_color};">{bin_details['note']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        if len(detections) > 1:
            with st.expander("All detected items in this image"):
                table_rows = []
                for d in detections:
                    bd = get_bin_details(d["class"])
                    table_rows.append(
                        {
                            "Class": d["class"],
                            "Confidence (%)": round(d["confidence"] * 100, 1),
                            "Recommended Bin": bd["bin_full_name"] if bd else "Not mapped",
                        }
                    )
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Save to history
        buffer = io.BytesIO()
        thumb = image_source.copy()
        thumb.thumbnail((160, 160))
        thumb.save(buffer, format="JPEG")

        st.session_state.history.insert(
            0,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "thumbnail": buffer.getvalue(),
                "class": top_detection["class"],
                "confidence": round(top_detection["confidence"] * 100, 1),
                "bin": bin_details["bin_full_name"] if bin_details else "Not mapped",
                "note": bin_details["note"] if bin_details else "",
            },
        )


# --------------------------------------------------------------------------------
# HISTORY SECTION
# --------------------------------------------------------------------------------

st.markdown("---")
st.subheader("Upload History")

if not st.session_state.history:
    st.info("No images have been analysed yet in this session.")
else:
    for record in st.session_state.history:
        hist_col1, hist_col2 = st.columns([1, 4])
        with hist_col1:
            st.image(record["thumbnail"], width=100)
        with hist_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**{record['class']}**  |  Confidence: {record['confidence']} percent")
            st.write(f"Recommended Bin: {record['bin']}")
            st.write(f"Note: {record['note']}")
            st.caption(record["timestamp"])
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    history_df = pd.DataFrame(
        [
            {
                "Timestamp": r["timestamp"],
                "Class": r["class"],
                "Confidence (%)": r["confidence"],
                "Recommended Bin": r["bin"],
            }
            for r in st.session_state.history
        ]
    )
    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download History as CSV",
        data=csv_data,
        file_name="waste_classification_history.csv",
        mime="text/csv",
    )
