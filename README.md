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
## File
- `acne_detection_and_cross_domain_classification.ipynb`  
  Colab notebook containing the full workflow.
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
- Dataset page: <https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/>
- Notebook downloads:
  - YOLO format (for YOLOv5)
  - COCO format (for Faster R-CNN)
### DermNet (Kaggle)
- Dataset: <https://www.kaggle.com/datasets/shubhamgoel27/dermnet>
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
