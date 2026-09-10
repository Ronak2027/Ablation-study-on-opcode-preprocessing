# Ablation Study on Opcode Preprocessing in Smart-Contract Vulnerability Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains an **ablation study** investigating the impact of EVM opcode operand value removal during preprocessing on smart-contract vulnerability detection models across four major vulnerability datasets.

---

## 📌 Research Question
> **How does removing opcode operand values (e.g. stripping inline PUSH constants/addresses while retaining opcode mnemonics) affect the predictive accuracy, F1-score, feature dimensionality, and computational efficiency of smart-contract vulnerability detection models?**

---

## 📊 Summary of Experimental Results

Both baseline (`With Opcode Values`) and ablation (`Without Opcode Values`) experiments were conducted under identical seeds (`42`), identical 80/20 train/test splits, identical `TfidfVectorizer`, and identical `DecisionTreeClassifier` hyperparameters.

| Dataset | Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Vocab Size | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Etherlock** | With Values (Baseline) | **96.89%** | **72.58%** | **67.16%** | **69.77%** | **0.8287** | 65,372 | 34.24s |
| | Without Values (Ablation) | 96.65% | 71.93% | 61.19% | 66.13% | 0.7992 | 40,774 | **17.97s** |
| | *Change (Ablation - Baseline)* | *-0.24 pp* | *-0.65 pp* | *-5.97 pp* | *-3.64 pp* | *-0.0294* | *-37.6%* | **1.9x faster** |
| **Block Dependency** | With Values (Baseline) | 95.41% | 0.00% | 0.00% | 0.00% | 0.4913 | 65,273 | 33.60s |
| | Without Values (Ablation) | **95.51%** | 0.00% | 0.00% | 0.00% | **0.4918** | 38,259 | **16.67s** |
| | *Change (Ablation - Baseline)* | *+0.10 pp* | *0.00 pp* | *0.00 pp* | *0.00 pp* | *+0.0005* | *-41.4%* | **2.0x faster** |
| **Integer Underflow / Overflow** | With Values (Baseline) | 81.99% | 47.77% | 36.95% | 41.67% | 0.6422 | 65,514 | 32.65s |
| | Without Values (Ablation) | **83.10%** | **51.92%** | **39.90%** | **45.13%** | **0.6606** | 41,158 | **13.92s** |
| | *Change (Ablation - Baseline)* | *+1.11 pp* | *+4.15 pp* | *+2.96 pp* | *+3.46 pp* | *+0.0184* | *-37.2%* | **2.3x faster** |
| **Reentrancy** | With Values (Baseline) | 90.94% | 34.62% | **38.57%** | **36.49%** | **0.6665** | 65,380 | 50.34s |
| | Without Values (Ablation) | **92.57%** | **43.14%** | 31.43% | 36.36% | 0.6421 | 39,337 | **17.75s** |
| | *Change (Ablation - Baseline)* | *+1.64 pp* | *+8.52 pp* | *-7.14 pp* | *-0.12 pp* | *-0.0243* | *-39.8%* | **2.8x faster** |

---

## 🔍 Key Insights

1. **Dimensionality & Efficiency**: Removing PUSH operand values consistently reduced TF-IDF feature vocabulary size by **37%–41%**, leading to **1.9x to 2.8x faster model training times**.
2. **Integer Underflow/Overflow**: Removing operand values improved performance across **all metrics** (+3.46 pp F1-score increase), confirming that stripping contract-specific constants/addresses reduces high-entropy noise for arithmetic pattern learning.
3. **Reentrancy**: Precision improved by **+8.52 pp** (reducing false positives from 51 to 29).
4. **Etherlock**: Operand values carry slight signal for contract locking, yielding a minor drop in recall (-5.97 pp) when stripped.

---

## 📁 Repository Structure

```
Solidity-Vuln-Detection/
├── dataset/                        # Smart contract EVM bytecode CSV datasets
│   ├── etherlock.csv
│   ├── block_dependancy.csv
│   ├── integer.csv
│   └── reentrancy.csv
├── backend/                        # Preprocessing & execution scripts
│   ├── preprocessing.py            # EVM bytecode opcode parser & operand stripper
│   ├── experiment.py               # Automated reproducible ablation test harness
│   ├── main.py                     # FastAPI web server
│   └── *.joblib                    # Pre-trained decision tree models
├── results/                        # Experimental output directory
│   ├── ablation_study_report.md    # Detailed Markdown research report
│   ├── metrics_comparison.csv      # CSV table with metric deltas
│   ├── experiment_summary.json     # Raw metrics & confusion matrices
│   ├── f1_comparison.png           # F1-Score bar chart
│   ├── accuracy_comparison.png     # Accuracy bar chart
│   ├── precision_comparison.png    # Precision bar chart
│   ├── recall_comparison.png       # Recall bar chart
│   └── confusion_matrices.png      # 4x2 Confusion matrices grid
├── frontend/                       # HTML frontend web interface
├── README.md                       # Main repository documentation
└── .gitignore
```

---

## 🚀 Quick Start & Installation

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Ronak2027/Ablation-study-on-opcode-pre-processing.git
cd Ablation-study-on-opcode-pre-processing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install pandas scikit-learn==1.4.1.post1 matplotlib seaborn joblib
```

### 2. Run the Full Ablation Experiment
To re-run both baseline and ablation experiments across all four datasets and generate figures/tables:
```bash
python backend/experiment.py
```
Output files will be automatically generated and updated under `results/`.

---

## 🛠️ Usage: Preprocessing Switch

You can use the EVM opcode preprocessing parser in your own smart contract analysis pipelines:

```python
from backend.preprocessing import preprocess_bytecode

raw_bytecode = "608060405234801561001057600080fd5b50610415"

# Baseline: Retain opcode values
tokens_with_values = preprocess_bytecode(raw_bytecode, use_opcode_values=True)
# Output: '60 80 60 40 52 34 80 15 61 00 10 57 60 00 80 fd 5b 50 61 04 15'

# Ablation: Remove PUSH operand values
tokens_without_values = preprocess_bytecode(raw_bytecode, use_opcode_values=False)
# Output: '60 60 52 34 80 15 61 57 60 80 fd 5b 50 61'
```

---

## 📜 Citation & Report
For detailed methodology, dataset breakdowns, confusion matrices, and threat validity analysis, refer to the full report in [`results/ablation_study_report.md`](results/ablation_study_report.md).
