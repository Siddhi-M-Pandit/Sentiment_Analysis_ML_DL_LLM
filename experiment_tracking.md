# 🎯 Experiment Tracking Setup

This document explains how experiment tracking is configured in this research project using **TensorBoard** and **Weights & Biases (wandb)**.

---
---

## 📌 1. TensorBoard Setup

**TensorBoard** is used to track training metrics and visualize deep learning experiments (CNN, BiLSTM models).

### ✅ How to Enable TensorBoard Logging

In your TensorFlow/Keras training script:
```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Create a unique log directory
log_dir = "results/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Add callback when training
model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_val, y_val),
          callbacks=[tensorboard_callback])
```

### 🚀 Running TensorBoard

On the **campus server terminal**:
```bash
tensorboard --logdir=results/logs --port=6006
```

On your **local machine** (to access dashboard):
```bash
ssh -N -L 6006:localhost:6006 smpandit@cscigpu.csuchico.edu
```
Then open:
```
http://localhost:6006
```

---

## 📌 2. Weights & Biases (wandb) Setup

**wandb** is used to log and visualize:
- Training & validation metrics
- Hyperparameters
- Model artifacts
- Experiment comparisons

### ✅ One-time setup on server
Run:
```bash
wandb login
```
Paste your API Key from https://wandb.ai/authorize

---

### 🟢 How to log experiments

#### For Deep Learning (Keras):
```python
import wandb
from wandb.keras import WandbCallback

wandb.init(project="Sentiment_Analysis", name="BiLSTM_exp1")

model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_val, y_val),
          callbacks=[WandbCallback()])
```

#### For Machine Learning (SVM, Naive Bayes):
You can manually log results:
```python
import wandb

wandb.init(project="Sentiment_Analysis", name="SVM_exp1")
wandb.log({"accuracy": accuracy, "f1_score": f1})
wandb.finish()
```

---

## 📂 Folder Structure

Experiment results will be stored in:
```
results/
 ┣ logs/          → TensorBoard logs
 ┣ wandb/         → wandb run files (auto-created)
 ┗ metrics/      → CSV/JSON files with evaluation metrics
```

---

## ⭐️ Best Practices
- Create a new **wandb run name** for each model/experiment
- Use the same project name → `Sentiment_Analysis`
- Keep `results/logs/` folder clean and commit only evaluation summaries (not raw logs)

---

## ✅ Quick Commands Summary
| Task                                   | Command                                              |
|---------------------------------------|-----------------------------------------------------|
| Start TensorBoard                     | `tensorboard --logdir=results/logs --port=6006`     |
| Create SSH Tunnel for TensorBoard     | `ssh -N -L 6006:localhost:6006 smpandit@cscigpu.csuchico.edu` |
| Login to wandb                        | `wandb login`                                       |
| View wandb dashboard                  | https://wandb.ai                                    |

---

**All experiments will be tracked here for report & result analysis.**
