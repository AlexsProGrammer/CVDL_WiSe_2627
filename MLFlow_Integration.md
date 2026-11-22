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
