# 🎯 Research Project Environment Setup Guide
**Project:** Comparative Analysis of ML, DL & LLM Approaches for Sentiment Analysis of News & Social Media Datasets  
**Researcher:** Siddhi M. Pandit  
**Server:** csicgpu.csuchico.edu (Chico State Campus GPU Server)

---
---

## 1. Remote Connection
Get access to Chico State campus GPU server from ITSS directly / through your Professor.
Connect using Remote-SSH:
```bash
ssh <username>@csicgpu.csuchico.edu
```

---

## 2. Clone Repository
Clone the GitHub repository inside your home directory:
```bash
git clone https://github.com/Siddhi-M-Pandit/Sentiment_Analysis_ML_DL_LLM.git
cd Sentiment_Analysis_ML_DL_LLM
```

---

## 3. Create & Activate Python Environment
Use Anaconda to create an isolated virtual environment:
```bash
conda create -n sentimentAnalysisENV python=3.10
conda activate sentimentAnalysisENV
```

---

## 4. Install Required Libraries
Run the following command to install the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow keras
pip install transformers
pip install jupyterlab tensorboard wandb
```

If you decide to use HuggingFace datasets later:
```bash
pip install datasets
```

---

## 5. Verify Installation
Check Python version & installed libraries:
```bash
python --version
pip list
```

---

## 6. Add Environment to Jupyter Kernel (Optional, but Recommended)
So that you can select this environment in Jupyter notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name=sentimentAnalysisENV
```

---

## 7. Running JupyterLab (on server)
To start JupyterLab:
```bash
jupyter lab --no-browser --port=8888
```

On your local terminal, run:
```bash
ssh -N -L 8888:localhost:8888 <username>@csicgpu.csuchico.edu
```
Then open → `http://localhost:8888` in your browser.

---

## 8. Folder Structure Overview
```
Sentiment_Analysis_ML_DL_LLM/
 ┣ 📂 data              # Raw & preprocessed datasets
 ┣ 📂 models            # Saved ML/DL/LLM models
 ┣ 📂 notebooks         # Jupyter notebooks for analysis
 ┣ 📂 results           # Evaluation reports, charts
 ┣ 📂 scripts           # Python scripts for training, evaluation
 ┣ 📄 README.md         
 ┣ 📄 requirements.txt  # Auto-generated after full setup
 ┗ 📄 setup_instructions.md
```

---

## 9. Save Environment Dependencies (Optional)
Once all packages are installed:
```bash
pip freeze > requirements.txt
```

---
