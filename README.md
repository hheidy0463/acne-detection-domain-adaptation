# Acne Detection and Cross-Domain Classification

This project has two parts:

1. **Part 1 (Detection on ACNE04):**
  - Train and compare **YOLOv5** and **Faster R-CNN** for acne lesion detection.
  - Report **mAP, Precision, Recall, IoU**.
  - Visualize predicted bounding boxes.
2. **Part 2 (Cross-domain Classification):**
  - Train classifier on ACNE04-derived patches (acne vs non-acne).
  - Evaluate transfer to DermNet (binary acne vs non-acne).
  - Apply domain adaptation (augmentation, histogram matching, pseudo-labeling).
  - Report **Accuracy, F1, AUROC** and Grad-CAM visualizations.

---

## Submission Requirement


| Requirement                                                        | File(s) in this repo                                                     |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Source code (organized, runnable)                                  | `acne_detection_and_cross_domain_classification.ipynb`                   |
| How to run                                                         | `README.md` (this file)                                                  |
| 1–2 page write-up (models, preprocessing, training/eval, findings) | `report.pdf` (compiled from `report.tex`; figures live under `figures/`) |


---

## Before you run

- **Platform:** The notebook is written for **Google Colab** (`/content/...` paths). It will **not** run end-to-end on your laptop without editing paths and install steps.
- **GPU:** Use **Runtime → Change runtime type → GPU**. Training is much slower on CPU and may time out on free Colab.
- **Roboflow:** You need an API key for ACNE04. Set environment variable `ROBOFLOW_API_KEY` in Colab (**Secrets** or **%env**), or the notebook will prompt you (paste when asked).
- **Kaggle / DermNet:** `kagglehub.dataset_download(...)` needs you to be logged into Kaggle. On Colab, typical options are:
  - Upload `kaggle.json` to `~/.kaggle/` and set permissions, **or**
  - Use a **Kaggle API token** / `kagglehub` login flow as in [kagglehub docs](https://github.com/Kaggle/kagglehub) (if download fails, this is usually why).
- **Run order:** Run cells **top to bottom** once per fresh runtime. Skipping training cells will leave variables undefined downstream.

---

## Environment

Recommended: **Google Colab (GPU runtime)**.
Main dependencies are installed in-notebook:

- `facenet-pytorch`
- `torchmetrics`
- `roboflow`
- `grad-cam`
- `numpy>=2.0.0,<2.1.0`
Additional required packages:
- `kagglehub`
- `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `Pillow`, `pyyaml`

---

## Dataset Setup

### ACNE04 (Roboflow)

- Dataset page: [https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/](https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/)
- Notebook downloads:
  - YOLO format (for YOLOv5)
  - COCO format (for Faster R-CNN)

### DermNet (Kaggle)

- Dataset: [https://www.kaggle.com/datasets/shubhamgoel27/dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)
- Notebook downloads via `kagglehub`.

---

## How to Run

Run sections in order from top to bottom:

1. **Setup + imports**
2. **Download ACNE04 and DermNet**
3. **Part 1**
  - Train YOLOv5
  - Train Faster R-CNN
  - Evaluate detection metrics
  - Visualize bounding box predictions
  - References cell (model selection justification)
4. **Part 2**
  - Create ACNE04 patches (positive/negative)
  - Train baseline classifier
  - Build DermNet binary test set (`acne` vs `non_acne`)
  - Evaluate baseline
  - Run adaptation experiments:
    - augmentation
    - histogram matching
    - pseudo-labeling (20 unlabeled DermNet train samples)
  - Evaluate all checkpoints
  - Generate Grad-CAM visualizations
  - Reflection on domain transfer results

---

## Key Outputs

Typical saved artifacts include:

- Detection models:
  - `faster_rcnn_acne.pth`
  - YOLO weights under `/content/runs/**/weights/best.pt`
- Classification models:
  - `classifier_acne.pth`
  - `classifier_adapted.pth`
  - `classifier_histmatch.pth`
  - `classifier_pseudolabel.pth`
- Visualizations:
  - `yolo_predictions.png`
  - `faster_rcnn_predictions.png`
  - `gradcam_dermnet_cls2.png`
  - `gradcam_dermnet_cls3.png`
  - `gradcam_dermnet_cls4.png`

---

## Notes

- This is a Colab notebook and uses `subprocess` to run shell commands via a `run_cmd` helper.
- For security best practice, set `ROBOFLOW_API_KEY` as an environment variable.
- Re-run evaluation cells after training to refresh final numbers used in the report.

---

## Assignment coverage

**Part 1 — ACNE04 detection**

- Two detectors (**YOLOv5** + **Faster R-CNN**), classic vs modern single-stage.
- Metrics: **mAP**, **Precision / Recall**, **IoU** (including matched IoU summary where implemented).
- **Bounding-box visualizations** on sample test images.
- **References** in the notebook justify model choices (Faster R-CNN, YOLO line, COCO/mAP conventions).

**Part 2 — Cross-domain classification**

- **Patches** from ACNE04 (positive / negative crops).
- **DermNet** binary test (**acne** vs **non-acne**); uses full test set for evaluation.
- **Domain adaptation:** face-pretrained network + stronger augmentation, **histogram matching**, **pseudo-labeling** on 20 unlabeled train samples (per prompt).
- Metrics: **Accuracy, F1, AUROC** on DermNet; **Grad-CAM** on 10+ test images (three adapted checkpoints).
- **Reflection** markdown summarizes transfer quality and what helped or hurt.

