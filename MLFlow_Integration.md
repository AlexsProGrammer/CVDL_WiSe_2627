# MLFlow

## Clean & Efficient PyTorch Integration

Never hardcode URLs or passwords inside your Python files. Because your GitHub repo is **public**, put credentials in a local `.env` file and add `.env` to `.gitignore` on Day 1.

### Local Setup (Each Teammate's Machine)

1. Install matching client packages:
```bash
pip install mlflow==2.17.2 python-dotenv

```


2. Create a local **`.env`** file (ignored by Git) with each person's own credentials:
```env
MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com
MLFLOW_TRACKING_USERNAME=alice
MLFLOW_TRACKING_PASSWORD=your-secret-password

```


MLflow automatically reads `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` from the environment—you don't even have to pass passwords into `mlflow.set_tracking_uri()`.

### Compact Training Loop Pattern (`train.py`)

```python
import os
import subprocess
import torch
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv

load_dotenv()  # Loads MLFLOW_TRACKING_URI, USERNAME, PASSWORD from .env


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def train(model, train_loader, val_loader, optimizer, criterion, config):
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config["run_name"]):
        # 1. Log hyperparameters + exact Git commit & author
        mlflow.log_params(config)
        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "author": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        })

        best_val_f1 = 0.0

        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

            # 2. Stream scalar curves live to your VPS
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1},
                step=epoch,
            )

            # 3. Save & upload best model weights whenever validation improves
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.log_artifact("best_model.pth", artifact_path="checkpoints")

        # 4. Log evaluation plots (confusion matrix, failure cases) at the end
        fig = plot_confusion_matrix(model, val_loader)
        fig.savefig("confusion_matrix.png", bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

```

---

## Beyond Loss & Accuracy: Everything Worth Tracking

MLflow is a complete experiment archive, not just a loss logger. Tracking the items below will give you ready-made material for both your **2-page pre-document** and **8-page final report**:

* **Model Weights (`best_model.pth`):** Uploading `state_dict()` checkpoints with `mlflow.log_artifact(...)` means your VPS becomes your shared model registry. If Teammate B trains a model overnight on their GPU, Teammate C can click that run in the MLflow UI, download `best_model.pth`, and run error analysis on it without sending multi-hundred-megabyte files over Discord.
* **Diagnostic Plots (`artifact_path="plots"`):** Log confusion matrices, precision-recall curves, and grids of **worst misclassified test images** (with true vs. predicted labels). Putting "Before vs. After" failure cases in your 8-page paper is one of the strongest ways to prove systematic engineering thought.
* **Hyperparameters & Architecture Choices (`log_params`):** Log learning rate, batch size, optimizer, weight decay, scheduler, backbone name, dropout rate, and image resolution. In the MLflow UI, you can select 10 runs and click **Compare** to get an automatic parallel-coordinates chart or scatter plot showing which parameter actually drove your score up.
* **Reproducibility Tags (`set_tags`):** Logging the short `git_commit` hash and random seed (`seed=42`) lets you trace any result in your final report back to the exact version of your code that produced it.
* **Dataset Split Reference:** Log the filename or hash of your locked `train_split.csv` / `val_split.csv` as a parameter so you have proof that every run in your ablation table was evaluated on identical validation data.


---

---

# Metadata + Infos Saved in MLFlow

### 1. Adding Info in Python Code (Best for Filtering & Ablation Tables)

In Python, split your metadata across **Parameters** (immutable experimental settings), **Tags** (flexible labels & notes), and **Text Artifacts** (exact PyTorch architecture):

```python
with mlflow.start_run(run_name="v2-resnet-with-attention"):
    # 1. Architectural switches & version numbers (become filterable table columns)
    mlflow.log_params({
        "model_version": "v2",
        "backbone": "resnet18",
        "use_attention": True,
        "attention_heads": 4,
        "lr": 1e-3,
    })

    # 2. Pre-fill the GUI's Description / Notes box directly from code
    mlflow.set_tag("mlflow.note.content", "Added CBAM spatial attention after block 3 to focus on foreground edges.")

    # 3. Save the exact PyTorch layer structure as a readable text file
    mlflow.log_text(str(model), "model_architecture.txt")

```

* **Why put `use_attention` and `model_version` in `log_params`?** In the MLflow UI, parameters become dedicated columns. You can type `params.use_attention = "True"` in the search bar (visible in your screenshot) to instantly filter runs, or select a `use_attention=False` run and a `use_attention=True` run side-by-side to generate an ablation comparison chart.
* **Why `mlflow.log_text(str(model), "model_architecture.txt")`?** Two months from now, when writing the methodology section of your 8-page paper, you won't have to guess what channel dimensions or dropout rates `v2` had—the exact PyTorch summary is saved inside that run's **Artifacts** tab.

---

### 2. Editing Info Directly in the Web GUI (Best for Post-Run Takeaways)

Once a run finishes and you look at its loss curves or confusion matrix, click on the **Run Name** in the table to open its detail page:

* **Description / Notes Box:** Right at the top of the run page is an editable **Description** panel (backed by the same `mlflow.note.content` tag). Click **Edit** and write Markdown notes about what you observed—for example:
> *Validation F1 jumped from 0.74 to 0.81 with attention, though training time doubled. Keep for final model.*


* **Tags Section:** Further down the detail page, you can add or modify custom key-value tags in the browser anytime—such as `report_status = "include_in_table_1"` or `issue = "overfitting"`.
* **Rename Run:** Click the three dots (`⋮`) next to any run to rename it (e.g., adding `[BASELINE]` or `[BEST]` to the front of the name) without affecting its logged metrics.

*(Note: **Parameters** cannot be edited in the GUI once logged—MLflow locks them intentionally so nobody can accidentally alter the record of how a model was actually trained—while **Run Names**, **Tags**, and **Descriptions** remain editable at any time.)*

---

### 3. Workflow Tip for Your 2-Page & 8-Page Papers

When you are ready to build the results tables for your papers:

1. Filter your runs in the `Experiments` table (e.g., by `tags.report_status = "final"`).
2. Click **Columns** in the UI to show only `model_version`, `use_attention`, `lr`, and `val_f1`.
3. Click the **Download CSV** button (in the `⋮` menu above the table) to export all your filtered runs into a clean spreadsheet you can paste straight into your paper.
