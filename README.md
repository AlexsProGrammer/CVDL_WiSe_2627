# CVDL_WiSe_2627
LMU Praktikum CV&DL Group Project


# Tools

| Tool | Desc | Notes |
|---|---|---|
| Nextcloud | FileShare | alles was nicht getracked werden muss bzw >10MB ist |
| GitHub | Codebase | |
| Discord | Communication | Questions, Features, Graphics of Training, ... |
| MLFlow | Weigths & Biases | Speicher/Management/Vergleich von Model durchläufen |


# Workflow

1. Clone Repo localy
2. Select Branch (Never work on main, always check that you are on the correct Branch)
3. Implement Feature by hand
4. Commit regularly to the branch (Do this when ever a small feature is working)
5. Only if features are tested and working push to main 


# General

- Create running simple 3 CNN baseline end of Week 1 (over fitted model)
  - this is to compare later builds with a base line
  - this also can be used to create something for the 2 page docs
  - test for our data loader and pipeline to check if everything works like intended
- Evaluation Metric (Accuracy/Macro-F1 Score/IoU)
- Lock Train/Val/Test Splits
  - use fixed seed
  - this is needed so we can compare the individual models versions to each other
  

# Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
deactivate
```

```bash
pip install -r requirements.txt
```


# Folder Structure

```
cvdl_wise_2627/
├── .gitignore                 # Verify data/, weights/, mlruns/, .venv/, *.pt, *.zip are ignored
├── README.md                  # Quickstart setup + Git branch rules + dataset download URLs
├── requirements.txt           # Pinned core packages (torch, torchvision, pandas, scikit-learn, tqdm, mlflow, pillow)
├── MLFlow_Integration.md      # MLFlow Guide
├── inference.py               # Provided evaluation harness (kept intact)
├── scripts/
│   ├── inspect_val.py         # Validation EDA script
│   ├── build_dataset.py       # Raw downloads → dataset_v1 + dataset_mini
│   └── train.py               # Stub for your unified training & MLflow runner
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Stub for Member 2 (Dataset + DataLoader)
│   ├── detector.py            # Stub for Member 4 (Largest cat/dog bbox crop)
│   ├── pipeline.py            # Stub for You (Combines detector + classifier + reject logic)
│   ├── xai.py                 # Stub for Phase 3 (Grad-CAM & saliency plots)
│   └── models/
│       ├── __init__.py
│       ├── scratch_cnn.py     # Stub for Member 3 (Mandatory from-scratch baseline)
│       └── backbone.py        # Stub for Phase 2 (Pretrained ResNet/ConvNeXt wrappers)
└── report/                    # Optional: unpacked LaTeX template for collaborative writing
```