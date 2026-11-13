import io
import requests
import streamlit as st

from PIL import Image, ImageDraw


# -----------------------------------------
# 1. Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="VisionPort",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------
# 2. Sidebar
# -----------------------------------------

st.sidebar.title("VisionPort Controls")

st.sidebar.write("Upload a traffic image")

API_URL = st.sidebar.text_input(
    "Backend URL",
    "http://127.0.0.1:8000"
).rstrip("/")

uploaded_file = st.sidebar.file_uploader(
    "Choose IDD image",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.divider()

conf = st.sidebar.slider(
    "Confidence Threshold",
    0.10, 0.90, 0.35, 0.05
)

iou = st.sidebar.slider(
    "IoU Threshold",
    0.10, 0.90, 0.50, 0.05
)

# Exported model has fixed 640x640 input
img_size = st.sidebar.selectbox(
    "Model Input Image Size",
    [640]
)

st.sidebar.caption(
    "This exported YOLOv8n model uses a fixed 640x640 input."
)


# -----------------------------------------
# 3. Backend Connection
# -----------------------------------------

st.sidebar.divider()

backend_status = False
device = "Unknown"

try:
    health = requests.get(
        f"{API_URL}/",
        timeout=5
    )

    health.raise_for_status()

    health_data = health.json()

    backend_status = True
    device = health_data.get("device", "Unknown")

    st.sidebar.success("Backend connected")

    st.sidebar.write(
        f"Execution Device: **{device}**"
    )

    st.sidebar.caption(
        "Available devices: " +
        ", ".join(health_data.get("available_devices", []))
    )

except requests.RequestException:

    st.sidebar.error(
        "FastAPI backend is not running."
    )


# -----------------------------------------
# 4. Main Page
# -----------------------------------------

st.title("🚦 VisionPort v0.1")

st.write(
    "GPU-Accelerated Indian Traffic Object Detection"
)

st.caption(
    "YOLOv8n + OpenVINO | Intel Iris Xe GPU / CPU"
)

st.divider()


# -----------------------------------------
# 5. Image Processing
# -----------------------------------------

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    if not backend_status:

        st.error(
            "Start the FastAPI backend before detecting objects."
        )

    else:

        with st.spinner(
            "Detecting traffic objects using OpenVINO..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/detect",

                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    },

                    params={
                        "conf": conf,
                        "iou": iou
                    },

                    timeout=120
                )

                response.raise_for_status()

                data = response.json()

            except requests.RequestException as error:

                st.error(
                    f"Detection request failed: {error}"
                )

                st.stop()

        detections = data["detections"]

        # ---------------------------------
        # 6. Draw Bounding Boxes
        # ---------------------------------

        result_image = image.copy()

        draw = ImageDraw.Draw(result_image)

        for item in detections:

            x1, y1, x2, y2 = item["box_xyxy"]

            name = item["class"]

            confidence = item["confidence"] * 100

            label = f"{name} {confidence:.1f}%"

            # Bounding box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="lime",
                width=3
            )

            # Label box with padding
            text_box = draw.textbbox(
                (x1, y1),
                label
            )

            padded_box = [
                text_box[0] - 2,
                text_box[1] - 2,
                text_box[2] + 2,
                text_box[3] + 2
            ]

            draw.rectangle(
                padded_box,
                fill="black"
            )

            draw.text(
                (x1, y1),
                label,
                fill="lime"
            )

        # ---------------------------------
        # 7. Display Images
        # ---------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Detection Result")

            st.image(
                result_image,
                use_container_width=True
            )

        st.divider()

        # ---------------------------------
        # 8. Detection Summary
        # ---------------------------------

        st.subheader("Detection Summary")

        total = data["num_detections"]

        timing = data["timing_ms"]

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total Objects Detected",
            total
        )

        metric2.metric(
            "Inference Device",
            data["device"]
        )

        metric3.metric(
            "Inference Time",
            f"{timing['inference']:.2f} ms"
        )

        st.divider()

        # ---------------------------------
        # 9. Object Counts
        # ---------------------------------

        st.subheader("Object Counts")

        counts = data["object_counts"]

        if counts:

            summary = [
                "| Object | Count | Average Confidence |",
                "| --- | ---: | ---: |"
            ]

            for name, count in counts.items():

                summary.append(
                    f"| {name} | {count} | "
                    f"{data['average_confidence'][name]:.3f} |"
                )

            st.markdown("\n".join(summary))

        else:

            st.warning("No objects detected.")

        # ---------------------------------
        # 10. Individual Detections
        # ---------------------------------

        st.subheader("Individual Detection Results")

        if detections:

            detection_table = [
                "| No. | Object | Confidence | Bounding Box |",
                "| ---: | --- | ---: | --- |"
            ]

            for index, item in enumerate(detections, 1):

                detection_table.append(
                    f"| {index} | {item['class']} | "
                    f"{item['confidence'] * 100:.2f}% | "
                    f"{item['box_xyxy']} |"
                )

            st.markdown("\n".join(detection_table))

        # ---------------------------------
        # 11. Performance Information
        # ---------------------------------

        st.subheader("Performance Information")

        st.json(timing)

        # ---------------------------------
        # 12. Download Result
        # ---------------------------------

        output = io.BytesIO()

        result_image.save(
            output,
            format="PNG"
        )

        st.download_button(
            "Download Detection Image",
            data=output.getvalue(),
            file_name="visionport_detection.png",
            mime="image/png"
        )

else:

    st.info(
        "Upload an IDD traffic image using the sidebar "
        "to begin object detection."
    )

    st.markdown(
        """
        ### VisionPort Features

        - YOLOv8n object detection
        - OpenVINO GPU acceleration
        - Intel Iris Xe support
        - Confidence score visualization
        - Object counting
        - Bounding box visualization
        """
    )