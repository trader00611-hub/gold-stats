"""DETR object detection with scene narration and VITS speech in Gradio."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

from scene_description import describe_scene

DETR_MODEL_ID = "facebook/detr-resnet-50"
VITS_MODEL_ID = "facebook/mms-tts-eng"


@lru_cache(maxsize=1)
def get_detector() -> Any:
    """Load DETR once and reuse it across requests."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("object-detection", model=DETR_MODEL_ID, device=device)


@lru_cache(maxsize=1)
def get_synthesizer() -> Any:
    """Load the English VITS text-to-speech pipeline once."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-to-speech", model=VITS_MODEL_ID, device=device)


def run_detection(image: Image.Image, threshold: float) -> list[dict[str, Any]]:
    detections = get_detector()(image.convert("RGB"))
    return [item for item in detections if float(item["score"]) >= threshold]


def annotate(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    width, height = rendered.size

    for item in detections:
        box = item["box"]
        x1 = max(0, min(width - 1, int(box["xmin"])))
        y1 = max(0, min(height - 1, int(box["ymin"])))
        x2 = max(0, min(width - 1, int(box["xmax"])))
        y2 = max(0, min(height - 1, int(box["ymax"])))
        label = f"{item['label']} {float(item['score']):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="#19d3a2", width=3)
        text_box = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        draw.rectangle(text_box, fill="#10231f")
        draw.text((x1, max(0, y1 - 18)), label, fill="white", font=font)
    return rendered


def speech_audio(text: str) -> tuple[int, np.ndarray]:
    result = get_synthesizer()(text)
    audio = np.asarray(result["audio"], dtype=np.float32).squeeze()
    return int(result["sampling_rate"]), audio


def analyze_image(image: Image.Image | None, threshold: float):
    if image is None:
        return None, "Upload an image or take a webcam snapshot first.", None, []
    image = image.convert("RGB")
    detections = run_detection(image, threshold)
    description, object_rows = describe_scene(detections, image.size, return_rows=True)
    audio = speech_audio(description)
    rows = [
        [item["label"], round(float(item["score"]), 3), item["location"]]
        for item in object_rows
    ]
    return annotate(image, detections), description, audio, rows


def live_detect(image: Image.Image | None, threshold: float):
    """Fast visual-only webcam stream; speech is generated on the Analyze action."""
    if image is None:
        return None, "Waiting for a webcam frame…"
    image = image.convert("RGB")
    detections = run_detection(image, threshold)
    description = describe_scene(detections, image.size)
    return annotate(image, detections), description


with gr.Blocks(title="Object Detection with Audio Description") as demo:
    gr.Markdown(
        "# Object Detection with Audio Description\n"
        "DETR finds COCO objects, a deterministic scene planner describes their count and position, "
        "and VITS speaks the description."
    )
    with gr.Tab("Analyze image"):
        with gr.Row():
            input_image = gr.Image(
                label="Image (upload or webcam snapshot)",
                sources=["upload", "webcam"],
                type="pil",
            )
            output_image = gr.Image(label="Detections", interactive=False)
        threshold = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="Minimum confidence")
        analyze_button = gr.Button("Detect and describe", variant="primary")
        description = gr.Textbox(label="Audio description", lines=3, interactive=False)
        detection_table = gr.Dataframe(
            headers=["Object", "Confidence", "Position"],
            datatype=["str", "number", "str"],
            label="Detected objects",
            interactive=False,
        )
        audio_output = gr.Audio(label="Generated speech", type="numpy", autoplay=False)
        analyze_button.click(
            analyze_image,
            inputs=[input_image, threshold],
            outputs=[output_image, description, audio_output, detection_table],
        )

    with gr.Tab("Live webcam"):
        gr.Markdown(
            "Live mode annotates frames and updates the object summary. Use Analyze image to generate "
            "speech for a selected frame; synthesizing audio on every frame would add latency."
        )
        with gr.Row():
            webcam = gr.Image(
                label="Webcam stream",
                sources=["webcam"],
                streaming=True,
                type="pil",
            )
            live_output = gr.Image(label="Live detections", interactive=False)
        live_summary = gr.Textbox(label="Current scene", lines=2, interactive=False)
        webcam.stream(
            live_detect,
            inputs=[webcam, threshold],
            outputs=[live_output, live_summary],
            stream_every=0.7,
            time_limit=60,
            concurrency_limit=1,
        )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
