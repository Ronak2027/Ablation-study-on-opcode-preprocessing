# Ablation Study on Opcode Preprocessing in Smart-Contract Vulnerability Detection

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains an **ablation study** investigating the impact of EVM opcode operand value removal during preprocessing on smart-contract vulnerability detection models across four major vulnerability datasets.

---

## 📌 Research Question
> **How does removing opcode operand values (stripping inline PUSH constants and addresses while retaining opcode instructions) affect predictive accuracy, F1-score, feature dimensionality, and computational efficiency in smart-contract vulnerability detection?**

---

## 📊 Summary of Experimental Results

Both baseline (`With Opcode Values`) and ablation (`Without Opcode Values`) experiments were conducted under identical seeds (`42`), identical 80/20 train/test splits, identical `TfidfVectorizer`, and identical `DecisionTreeClassifier` hyperparameters.

### Predictive Performance & Vocabulary Dimensionality

| Dataset | Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Vocab Size | Delta Vocab (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Etherlock** | With Values (Baseline) | **96.89%** | **72.58%** | **67.16%** | **69.77%** | **0.8287** | 65,372 | — |
| | Without Values (Ablation) | 96.65% | 71.93% | 61.19% | 66.13% | 0.7992 | 40,774 | **-37.6%** |
| | *Delta (Ablation - Baseline)* | *-0.24 pp* | *-0.65 pp* | *-5.97 pp* | *-3.64 pp* | *-0.0294* | *-24,598* | |
| **Block Dependency** | With Values (Baseline) | 95.41% | 0.00% | 0.00% | 0.00% | 0.4913 | 65,273 | — |
| | Without Values (Ablation) | **95.51%** | 0.00% | 0.00% | 0.00% | **0.4918** | 38,259 | **-41.4%** |
| | *Delta (Ablation - Baseline)* | *+0.10 pp* | *0.00 pp* | *0.00 pp* | *0.00 pp* | *+0.0005* | *-27,014* | |
| **Integer Underflow / Overflow** | With Values (Baseline) | 81.99% | 47.77% | 36.95% | 41.67% | 0.6422 | 65,514 | — |
| | Without Values (Ablation) | **83.10%** | **51.92%** | **39.90%** | **45.13%** | **0.6606** | 41,158 | **-37.2%** |
| | *Delta (Ablation - Baseline)* | *+1.11 pp* | *+4.15 pp* | *+2.96 pp* | *+3.46 pp* | *+0.0184* | *-24,356* | |
| **Reentrancy** | With Values (Baseline) | 90.94% | 34.62% | **38.57%** | **36.49%** | **0.6665** | 65,380 | — |
| | Without Values (Ablation) | **92.57%** | **43.14%** | 31.43% | 36.36% | 0.6421 | 39,337 | **-39.8%** |
| | *Delta (Ablation - Baseline)* | *+1.64 pp* | *+8.52 pp* | *-7.14 pp* | *-0.12 pp* | *-0.0243* | *-26,043* | |

---

### Computational Time Breakdown (TF-IDF vs Decision Tree Fitting)

| Dataset | Metric Component | Baseline Time (s) | Ablation Time (s) | Speedup Multiplier |
| :--- | :--- | :---: | :---: | :---: |
| **Etherlock** | TF-IDF Fitting Time | 47.28s | 23.47s | **2.01x faster** |
| | Decision Tree Fitting Time | 39.77s | 21.95s | **1.81x faster** |
| | **Total Pipeline Time** | **87.05s** | **45.41s** | **1.92x faster** |
| **Block Dependency** | TF-IDF Fitting Time | 38.35s | 16.92s | **2.27x faster** |
| | Decision Tree Fitting Time | 44.91s | 23.99s | **1.87x faster** |
| | **Total Pipeline Time** | **83.25s** | **40.91s** | **2.04x faster** |
| **Integer Underflow / Overflow** | TF-IDF Fitting Time | 49.38s | 17.84s | **2.77x faster** |
| | Decision Tree Fitting Time | 34.45s | 13.75s | **2.51x faster** |
| | **Total Pipeline Time** | **83.83s** | **31.59s** | **2.65x faster** |
| **Reentrancy** | TF-IDF Fitting Time | 43.48s | 18.77s | **2.32x faster** |
| | Decision Tree Fitting Time | 71.26s | 23.11s | **3.08x faster** |
| | **Total Pipeline Time** | **114.74s** | **41.88s** | **2.74x faster** |

---

## 🔍 Key Insights

1. **Dimensionality Reduction**: Stripping PUSH operands slashes TF-IDF feature vocabulary by **37.2% to 41.4%**.
2. **Computational Acceleration**:
   - The tree classifier alone trains **1.8x to 3.1x faster**.
   - TF-IDF feature extraction is **2.0x to 2.8x faster**.
   - Total end-to-end training pipeline achieves a **1.9x to 2.7x speedup**.
3. **Integer Underflow/Overflow**: Removing operand values improved performance across **all metrics** (+3.46 pp F1-score increase), confirming that stripping contract-specific constants/addresses reduces high-entropy noise for arithmetic pattern learning.
4. **Reentrancy**: Precision improved by **+8.52 pp** (reducing false positives from 51 to 29).
5. **Etherlock**: Operand values carry slight signal for contract locking, yielding a minor drop in recall (-5.97 pp) when stripped.

---

## 📁 Repository Structure

```
Ablation-study-on-opcode-preprocessing/
├── dataset/                        # Smart contract EVM bytecode CSV datasets
│   ├── etherlock.csv
│   ├── block_dependancy.csv
│   ├── integer.csv
│   └── reentrancy.csv
├── backend/                        # Preprocessing & execution scripts
│   ├── preprocessing.py            # EVM bytecode opcode parser & operand stripper
│   ├── experiment.py               # Automated reproducible ablation test harness
│   ├── main.py                     # FastAPI web server with dual-pipeline inference
│   ├── model_manifest.json         # Reconciled model metadata, vocabularies, & metrics
│   ├── models/                     # Saved models & fitted TF-IDF vectorizers
│   │   ├── *_with_operands_model.joblib
│   │   ├── *_with_operands_vectorizer.joblib
│   │   ├── *_without_operands_model.joblib
│   │   └── *_without_operands_vectorizer.joblib
│   └── *_decisiontree.joblib       # Baseline models & vectorizers
├── results/                        # Experimental outputs & figures
│   ├── ablation_study_report.md    # Detailed Markdown research report
│   ├── metrics_comparison.csv      # CSV table with metric deltas & timing breakdown
│   ├── model_manifest.json         # Copy of model manifest
│   ├── experiment_summary.json     # Raw metrics & confusion matrices
│   ├── training_time_breakdown.png # 3-panel TF-IDF vs DT vs Pipeline time comparison
│   ├── f1_comparison.png           # F1-Score bar chart
│   ├── accuracy_comparison.png     # Accuracy bar chart
│   ├── precision_comparison.png    # Precision bar chart
│   ├── recall_comparison.png       # Recall bar chart
│   └── confusion_matrices.png      # 4x2 Confusion matrices grid
├── frontend/                       # Interactive Web Application & Presentation Deck
│   ├── index.html                  # Full interactive analyzer, metrics viewer & slides
│   └── results.html                # Standalone results table
├── README.md                       # Main repository documentation
└── .gitignore
```

---

## 🚀 Running the Project & Presentation

### 1. Start the Backend API Server
Ensure dependencies are installed and run the FastAPI server:
```bash
python -m pip install fastapi uvicorn scikit-learn pandas joblib matplotlib seaborn py-solc-x
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The server will initialize:
- 4 Baseline models + vectorizers (`with_operands`)
- 4 Ablation models + vectorizers (`without_operands`)
- Endpoints: `http://localhost:8000/docs`, `/analyze`, `/manifest`, `/metrics`

### 2. Launch the Web Interface & Presentation
Open `frontend/index.html` in your web browser. The interface includes:
- 🔬 **Contract Analyzer**: Live contract scanning with side-by-side baseline vs ablation comparison, token reduction meter, and confidence scores.
- 📊 **Experimental Results**: Full interactive tables, vocabulary reduction figures, and confusion matrices.
- 📑 **Slide Presentation**: An integrated slide presentation deck designed for project defense and presentation (including Slide 9's isolated training time breakdown).

### 3. Re-running the Full Ablation Experiment
To re-run the complete benchmark and regenerate all plots and tables:
```bash
python backend/experiment.py
```
Output files will be automatically generated and updated under `results/`.
