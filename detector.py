
import time
import cv2
import numpy as np
import openvino as ov


# COCO class names
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class VisionPortDetector:

    def __init__(self, model_path="models/yolov8n.xml"):

        self.core = ov.Core()

        self.model = self.core.read_model(model_path)

        self.input_layer = self.model.input(0)

        shape = list(self.input_layer.shape)

        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(
                f"Expected NCHW model input, got {shape}"
            )

        self.input_height = int(shape[2])
        self.input_width = int(shape[3])

        if self.input_height != self.input_width:
            raise ValueError("This detector expects square input.")

        self.device = None
        self.compiled_model = None

        # Prefer Intel GPU, otherwise CPU
        if "GPU" in self.core.available_devices:

            try:
                self.compiled_model = self.core.compile_model(
                    self.model, "GPU"
                )
                self.device = "GPU"

            except Exception as error:
                print("GPU unavailable:", error)

        if self.compiled_model is None:

            self.compiled_model = self.core.compile_model(
                self.model, "CPU"
            )

            self.device = "CPU"

        print("VisionPort inference device:", self.device)

    @staticmethod
    def _add_trace_stage(stages, name, where, data_format, shape, start, note=""):
        stages.append({
            "stage": len(stages) + 1,
            "name": name,
            "where": where,
            "format": data_format,
            "shape": shape,
            "note": note,
            "ms": round((time.perf_counter() - start) * 1000, 2)
        })

    def preprocess(self, image, stages=None):

        letterbox_start = time.perf_counter()
        height, width = image.shape[:2]

        scale = min(
            self.input_width / width,
            self.input_height / height
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height)
        )

        canvas = np.full(
            (self.input_height, self.input_width, 3),
            114,
            dtype=np.uint8
        )

        left = (self.input_width - new_width) // 2
        top = (self.input_height - new_height) // 2

        canvas[
            top:top + new_height,
            left:left + new_width
        ] = resized

        if stages is not None:
            self._add_trace_stage(
                stages,
                "letterbox resize",
                "CPU / gateway",
                "numpy uint8 (H,W,C)",
                list(canvas.shape),
                letterbox_start,
                "aspect ratio kept, grey padding added"
            )

        tensor_start = time.perf_counter()
        tensor = canvas[:, :, ::-1]
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)
        tensor = tensor.astype(np.float32) / 255.0

        if stages is not None:
            self._add_trace_stage(
                stages,
                "to tensor",
                "CPU / gateway",
                "numpy float32 (N,C,H,W)",
                list(tensor.shape),
                tensor_start,
                "channels first, batch dimension added, scaled to 0-1"
            )

        return tensor, scale, left, top

    def predict(self, image, conf=0.35, iou=0.50, trace_stages=None):

        total_start = time.perf_counter()

        # Preprocessing
        start = time.perf_counter()

        tensor, scale, left, top = self.preprocess(image, trace_stages)

        preprocess_time = (
            time.perf_counter() - start
        ) * 1000

        # OpenVINO inference
        start = time.perf_counter()

        try:
            result = self.compiled_model([tensor])

        except Exception:
            if self.device != "GPU":
                raise

            print("GPU inference failed. Switching to CPU.")

            self.compiled_model = self.core.compile_model(
                self.model, "CPU"
            )

            self.device = "CPU"

            result = self.compiled_model([tensor])

        inference_time = (
            time.perf_counter() - start
        ) * 1000

        decode_start = time.perf_counter()
        raw_result = np.asarray(
            result[self.compiled_model.output(0)]
        )

        if trace_stages is not None:
            self._add_trace_stage(
                trace_stages,
                "model forward",
                f"OpenVINO / {self.device}",
                "numpy float32",
                list(raw_result.shape),
                start,
                "OpenVINO manages input transfer; transfer time is included here"
            )

        # Extract YOLO output
        start = decode_start
        output = raw_result[0]

        # YOLOv8 output: (84, 8400)
        if output.shape[0] < output.shape[1]:
            output = output.T

        if output.shape[1] != 84:
            raise ValueError(
                f"Unexpected YOLO output shape: {output.shape}"
            )

        boxes = output[:, :4]
        class_scores = output[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        valid = scores >= conf

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        height, width = image.shape[:2]

        candidates = []

        for box, score, class_id in zip(
            boxes, scores, class_ids
        ):

            cx, cy, bw, bh = box

            x1 = (cx - bw / 2 - left) / scale
            y1 = (cy - bh / 2 - top) / scale
            x2 = (cx + bw / 2 - left) / scale
            y2 = (cy + bh / 2 - top) / scale

            x1 = max(0, min(width, int(x1)))
            y1 = max(0, min(height, int(y1)))
            x2 = max(0, min(width, int(x2)))
            y2 = max(0, min(height, int(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            candidates.append({
                "class_id": int(class_id),
                "class": CLASSES[int(class_id)],
                "confidence": float(score),
                "box_xyxy": [x1, y1, x2, y2]
            })

        if trace_stages is not None:
            self._add_trace_stage(
                trace_stages,
                "decode and rescale boxes",
                "CPU / gateway",
                "numpy arrays",
                {
                    "raw_output": list(output.shape),
                    "candidates": len(candidates)
                },
                start,
                "confidence filtering, class selection, and mapping to original image pixels"
            )

        # Class-wise Non-Maximum Suppression
        nms_start = time.perf_counter()
        detections = []

        class_ids_present = set(
            item["class_id"] for item in candidates
        )

        for class_id in class_ids_present:

            group = [
                item for item in candidates
                if item["class_id"] == class_id
            ]

            nms_boxes = []

            for item in group:

                x1, y1, x2, y2 = item["box_xyxy"]

                nms_boxes.append([
                    x1, y1, x2 - x1, y2 - y1
                ])

            indices = cv2.dnn.NMSBoxes(
                nms_boxes,
                [item["confidence"] for item in group],
                conf,
                iou
            )

            if len(indices) > 0:

                for index in np.array(indices).flatten():

                    detections.append(group[int(index)])

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        postprocess_time = (
            time.perf_counter() - start
        ) * 1000

        if trace_stages is not None:
            self._add_trace_stage(
                trace_stages,
                "NMS",
                "CPU / gateway",
                "detection list",
                [len(detections), 6],
                nms_start,
                "class-wise OpenCV NMS; columns are x1, y1, x2, y2, confidence, class id"
            )

        total_time = (
            time.perf_counter() - total_start
        ) * 1000

        return {
            "detections": detections,
            "num_detections": len(detections),
            "device": self.device,
            "timing_ms": {
                "preprocess": round(preprocess_time, 2),
                "inference": round(inference_time, 2),
                "postprocess": round(postprocess_time, 2),
                "total": round(total_time, 2)
            }
        }
