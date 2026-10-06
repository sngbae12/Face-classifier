"""YOLO11n 사물 인식 웹 (Gradio + Ultralytics + OpenCV).

- 좌측: 이미지 업로드 -> 탐지 결과 이미지 표시
- 우측: 웹캠 스트리밍 -> 프레임마다 탐지, 바운딩 박스가 그려진 영상 실시간 표시
"""

import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as class_palette

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"  # YOLO11 nano. 파일이 없으면 최초 실행 시 자동 다운로드
CONF_THRESHOLD = 0.25  # 이 값 이상의 신뢰도만 표시
IMG_SIZE = 640  # 추론 입력 크기
STREAM_EVERY = 0.1  # 웹캠 프레임 전송 간격(초)

# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------
model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names  # {클래스 id: 이름}
COLORS = [class_palette(i, bgr=True) for i in range(len(CLASS_NAMES))]  # 클래스별 BGR 색상
_predict_lock = threading.Lock()  # 이미지/웹캠 요청이 동시에 들어와도 모델을 한 번에 하나씩 사용


def _text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """배경색 밝기에 따라 글자색(검정/흰색)을 고른다."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


# ---------------------------------------------------------------------------
# 영상 처리 (OpenCV)
# ---------------------------------------------------------------------------
def draw_detections(image_bgr: np.ndarray, result) -> np.ndarray:
    """탐지 결과를 바탕으로 바운딩 박스, 범주이름, 신뢰도를 이미지 위에 그린다."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    thickness = max(1, round(min(h, w) / 320))
    font_scale = max(0.4, min(h, w) / 1000)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cls = int(cls)
        color = COLORS[cls % len(COLORS)]
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"

        # 바운딩 박스
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)

        # 라벨 배경 + 글자 (박스 위에 공간이 없으면 박스 안쪽 위에 그림)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        outside = y1 - th - baseline >= 0
        top = y1 - th - baseline if outside else y1
        bottom = y1 if outside else y1 + th + baseline
        cv2.rectangle(image_bgr, (x1, top), (x1 + tw, bottom), color, cv2.FILLED)
        cv2.putText(
            image_bgr,
            label,
            (x1, bottom - baseline),
            font,
            font_scale,
            _text_color(color),
            thickness,
            cv2.LINE_AA,
        )

    return image_bgr


# ---------------------------------------------------------------------------
# 인식 실행 (Ultralytics)
# ---------------------------------------------------------------------------
def detect(image_rgb: np.ndarray | None) -> np.ndarray | None:
    """RGB 이미지(numpy)를 받아 사물을 탐지하고, 결과가 그려진 RGB 이미지를 돌려준다.

    이미지 업로드와 웹캠 프레임 모두 이 함수를 사용한다.
    """
    if image_rgb is None:
        return None

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Gradio(RGB) -> OpenCV/YOLO(BGR)
    with _predict_lock:
        result = model.predict(image_bgr, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
    annotated_bgr = draw_detections(image_bgr, result)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)  # OpenCV(BGR) -> Gradio(RGB)


# 첫 요청 지연을 줄이기 위한 워밍업
detect(np.zeros((480, 640, 3), dtype=np.uint8))

# ---------------------------------------------------------------------------
# UI (Gradio) - 좌측 50%: 이미지 업로드 / 우측 50%: 웹캠
# ---------------------------------------------------------------------------
# 웹캠 결과 프레임을 라이브 미리보기와 같은 방식(박스 안에 맞춤)으로 표시
CSS = "#webcam video + img { width: 100%; height: 100%; object-fit: contain; }"

with gr.Blocks(title="YOLO11n 사물 인식", fill_width=True) as demo:
    gr.Markdown("## YOLO11n 사물 인식")

    with gr.Row(equal_height=True):
        # 좌측: 이미지 입력
        with gr.Column(scale=1):
            gr.Markdown("### 이미지 입력")
            image_input = gr.Image(
                sources=["upload"],
                type="numpy",
                label="이미지 업로드",
                height=320,
            )
            image_output = gr.Image(
                label="탐지 결과",
                interactive=False,
                height=320,
            )

        # 우측: 웹캠 입력 (탐지 결과가 같은 화면에 실시간으로 표시됨)
        with gr.Column(scale=1):
            gr.Markdown("### 웹캠 입력")
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="웹캠 (실시간 탐지)",
                height=700,
                elem_id="webcam",
                webcam_options=gr.WebcamOptions(
                    constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}}
                ),
            )

    # 이미지가 업로드(또는 삭제)되면 탐지 실행
    image_input.change(detect, inputs=image_input, outputs=image_output)

    # 웹캠 프레임이 들어올 때마다 탐지 실행, 결과를 웹캠 화면에 바로 표시
    webcam.stream(detect, inputs=webcam, outputs=webcam, stream_every=STREAM_EVERY)

if __name__ == "__main__":
    demo.launch(css=CSS)
