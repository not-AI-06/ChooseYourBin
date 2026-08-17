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
or update MODEL_PATH below.
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO



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
    .stApp {
        background-color: #F4F6F5;
    }
    section[data-testid="stSidebar"] {
        background-color: #EDEFEE;
    }
    .bin-card {
        padding: 18px 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 10px;
    }
    .bin-card h3 {
        margin: 0 0 6px 0;
    }
    .bin-card p {
        margin: 0;
        font-size: 0.95rem;
    }
    .result-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 4px;
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
    st.sidebar.markdown(
        f"""
        <div class="bin-card" style="background-color:{info['color']};">
            <h3>{bin_name} Bin</h3>
            <p>{info['description']}</p>
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

input_method = st.radio("Choose input method", ["Upload Image", "Take Photograph"], horizontal=True)

image_source = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image of the waste item", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")
else:
    camera_file = st.camera_input("Take a photograph of the waste item")
    if camera_file is not None:
        image_source = Image.open(camera_file).convert("RGB")


# --------------------------------------------------------------------------------
# INFERENCE AND RESULTS
# --------------------------------------------------------------------------------

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


if image_source is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(image_source, use_container_width=True)

    with st.spinner("Analysing image..."):
        results = model.predict(source=np.array(image_source), conf=confidence_threshold, verbose=False)

    result = results[0]

    with col2:
        st.subheader("Detection Result")
        annotated = result.plot()  # returns BGR numpy array with boxes drawn
        annotated_rgb = annotated[:, :, ::-1]
        st.image(annotated_rgb, use_container_width=True)

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        st.warning("No waste item could be confidently identified in this image. Try a clearer photograph or lower the confidence threshold.")
    else:
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"class": class_name, "confidence": conf})

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        top_detection = detections[0]

        bin_details = get_bin_details(top_detection["class"])

        st.markdown("---")
        st.markdown(f"<div class='result-header'>Identified Waste: {top_detection['class']}</div>", unsafe_allow_html=True)
        st.write(f"Confidence: {top_detection['confidence'] * 100:.1f} percent")

        if bin_details is None:
            st.warning("No bin mapping is defined for this class yet.")
        else:
            st.markdown(
                f"""
                <div class="bin-card" style="background-color:{bin_details['bin_color']};">
                    <h3>{bin_details['bin_full_name']}</h3>
                    <p>{bin_details['note']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
    for i, record in enumerate(st.session_state.history):
        hist_col1, hist_col2 = st.columns([1, 4])
        with hist_col1:
            st.image(record["thumbnail"], width=100)
        with hist_col2:
            st.markdown(f"**{record['class']}**  |  Confidence: {record['confidence']} percent")
            st.write(f"Recommended Bin: {record['bin']}")
            st.write(f"Note: {record['note']}")
            st.caption(record["timestamp"])
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
