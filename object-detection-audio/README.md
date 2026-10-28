# AI-Based Object Detection with Audio Description

An interactive Python app that detects common objects with DETR, draws confidence-labelled bounding boxes, builds a concise description from the detected labels and positions, and speaks it with a VITS-based Hugging Face model. Gradio supports image upload, webcam snapshots, and a live visual webcam stream.

## Pipeline

1. DETR (`facebook/detr-resnet-50`) detects objects from the COCO label set.
2. The app filters by a confidence threshold and draws bounding boxes, class names, and scores.
3. `scene_description.py` groups labels into counts, locates detections on the left/center/right, and highlights the strongest detection. It describes only model outputs; it does not infer unsupported scene details.
4. VITS via MMS-TTS (`facebook/mms-tts-eng`) synthesizes the description as audio.
5. Gradio shows the annotated image, object table, text, and playable audio. Its live webcam tab updates visual detection and scene text; audio runs on demand to avoid synthesizing speech for every camera frame.

## Run locally

Python 3.10 or newer is recommended. Install a PyTorch build suitable for your CPU/GPU from the official PyTorch install selector, then install the rest of the requirements.

    cd object-detection-audio
    python -m venv .venv
    # Windows PowerShell
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python app.py

Open the local Gradio URL printed in the terminal. The DETR and TTS weights download from Hugging Face on first use; the initial request can take time and requires internet access. CPU inference works but is slower; CUDA is selected automatically when available.

## Models and licensing

- Detector: [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50), DETR trained on COCO. The model card lists Apache-2.0.
- Speech: [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng), a VITS architecture English model. The model card lists CC-BY-NC-4.0; that is a non-commercial license. Review both model licenses and their terms before redistribution or commercial use.
- Libraries: [Transformers DETR documentation](https://huggingface.co/docs/transformers/model_doc/detr), [Transformers MMS/VITS documentation](https://huggingface.co/docs/transformers/model_doc/mms), [Gradio streaming inputs](https://www.gradio.app/guides/streaming-inputs).

## Notes

- The app uses pretrained weights; it does not fine-tune DETR or VITS.
- DETR is a single-frame detector, not a multi-object tracker. “Live” means repeated webcam-frame inference, not persistent object IDs.
- COCO classes are limited to the categories covered by the checkpoint. Confidence scores are model scores, not calibrated probabilities.
- Use a local trusted environment. Uploaded images are processed by the local application; model weights are downloaded from Hugging Face.
