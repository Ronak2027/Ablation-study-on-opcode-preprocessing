# Final Report: Ablation Study on Opcode Preprocessing in Smart-Contract Vulnerability Detection

## 1. Objective
The objective of this study is to empirically evaluate the impact of opcode operand value removal during EVM bytecode preprocessing on the performance of an existing smart-contract vulnerability detection decision tree model across four distinct vulnerability datasets.

## 2. Research Question
**How does removing opcode operand values during preprocessing affect the performance of the existing smart-contract vulnerability detection model?**

---

## 3. Dataset Description
The experiment evaluated four smart-contract vulnerability datasets:
- **Etherlock**: 6,269 contracts (6,000 Non-vulnerable [0], 269 Vulnerable [1]).
- **Block Dependency**: 5,006 contracts (4,892 Non-vulnerable [0], 114 Vulnerable [1]).
- **Integer Underflow / Overflow**: 5,829 contracts (4,892 Non-vulnerable [0], 937 Vulnerable [1]).
- **Reentrancy**: 5,185 contracts (4,892 Non-vulnerable [0], 293 Vulnerable [1]).

Each CSV dataset contains:
- `contract_address`: Unique EVM contract address string.
- `bytecode`: Raw hex string of compiled Solidity EVM bytecode.
- `is_vulnerable`: Target binary class label (`0` or `1`).

---

## 4. Existing Preprocessing
The existing codebase ([`backend/main.py`](file:///d:/Solidity-Vuln-Detection/backend/main.py)) paired raw hexadecimal characters in steps of 2:
```python
def bytecode_to_tokens(bytecode: str) -> str:
    return ' '.join(bytecode[i:i+2] for i in range(0, len(bytecode), 2))
```
This raw byte tokenization processes both opcode bytes (e.g., `60` for PUSH1, `61` for PUSH2) and operand bytes indiscriminately.

---

## 5. "With Opcode Values" Methodology (Baseline)
- Retains both the opcode byte/instruction and its subsequent inline operand bytes in sequence.
- **Example**: `PUSH1 0x80 PUSH2 0x0415 MSTORE` -> `60 80 60 40 52`
- Serves as the benchmark baseline experiment (`USE_OPCODE_VALUES = True`).

---

## 6. "Without Opcode Values" Methodology (Ablation)
- Identifies EVM PUSH instructions (`0x60` to `0x7F` representing PUSH1 to PUSH32).
- Retains the PUSH opcode byte itself while stripping the subsequent $N$ operand bytes ($N = 1 ... 32$).
- **Example**: `PUSH1 0x80 PUSH2 0x0415 MSTORE` -> `60 61 52` (or `PUSH1 PUSH2 MSTORE`).
- Executed via `USE_OPCODE_VALUES = False`.

---

## 7. Model Configuration
- **Algorithm**: `sklearn.tree.DecisionTreeClassifier`
- **Parameters**:
  - `criterion`: `'gini'`
  - `max_depth`: `None`
  - `random_state`: `42`
  - `min_samples_split`: `2`
  - `min_samples_leaf`: `1`
- **Feature Extractor**: `TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b', ngram_range=(1, 2), min_df=1)`

---

## 8. Experimental Setup
- **Train / Test Split**: Fixed 80/20 train/test split with `random_state=42`.
- **Seed Control**: All data splits, vectorizers, and decision tree initializations used identical seed `42`.
- **Environment**: Python 3.12.13 with `scikit-learn==1.4.1.post1`.
- **Experimental Control**: Both experiments used identical dataset rows, identical target labels, identical train/test indices, and identical model hyperparameters. The **ONLY** variable was opcode operand retention (`USE_OPCODE_VALUES`).

---

## 9. Experimental Results Table

| Dataset | Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Vocab Size | Train Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Etherlock** | With Values (Baseline) | **96.89%** | **72.58%** | **67.16%** | **69.77%** | **0.8287** | 65,372 | 34.24s |
| | Without Values (Ablation) | 96.65% | 71.93% | 61.19% | 66.13% | 0.7992 | 40,774 | **17.97s** |
| | *Delta (Ablation - Baseline)* | *-0.24 pp* | *-0.65 pp* | *-5.97 pp* | *-3.64 pp* | *-0.0294* | *-37.6%* | *1.9x faster* |
| **Block Dependency** | With Values (Baseline) | 95.41% | 0.00% | 0.00% | 0.00% | 0.4913 | 65,273 | 33.60s |
| | Without Values (Ablation) | **95.51%** | 0.00% | 0.00% | 0.00% | **0.4918** | 38,259 | **16.67s** |
| | *Delta (Ablation - Baseline)* | *+0.10 pp* | *0.00 pp* | *0.00 pp* | *0.00 pp* | *+0.0005* | *-41.4%* | *2.0x faster* |
| **Integer Underflow / Overflow** | With Values (Baseline) | 81.99% | 47.77% | 36.95% | 41.67% | 0.6422 | 65,514 | 32.65s |
| | Without Values (Ablation) | **83.10%** | **51.92%** | **39.90%** | **45.13%** | **0.6606** | 41,158 | **13.92s** |
| | *Delta (Ablation - Baseline)* | *+1.11 pp* | *+4.15 pp* | *+2.96 pp* | *+3.46 pp* | *+0.0184* | *-37.2%* | *2.3x faster* |
| **Reentrancy** | With Values (Baseline) | 90.94% | 34.62% | **38.57%** | **36.49%** | **0.6665** | 65,380 | 50.34s |
| | Without Values (Ablation) | **92.57%** | **43.14%** | 31.43% | 36.36% | 0.6421 | 39,337 | **17.75s** |
| | *Delta (Ablation - Baseline)* | *+1.64 pp* | *+8.52 pp* | *-7.14 pp* | *-0.12 pp* | *-0.0243* | *-39.8%* | *2.8x faster* |

---

## 10. Confusion Matrices

```
Etherlock — With Values:
  [ True Negatives (TN): 1170  |  False Positives (FP): 17 ]
  [ False Negatives (FN): 22   |  True Positives  (TP): 45 ]

Etherlock — Without Values:
  [ True Negatives (TN): 1171  |  False Positives (FP): 16 ]
  [ False Negatives (FN): 26   |  True Positives  (TP): 41 ]

Block Dependency — With Values:
  [ True Negatives (TN): 956   |  False Positives (FP): 17 ]
  [ False Negatives (FN): 29   |  True Positives  (TP): 0  ]

Block Dependency — Without Values:
  [ True Negatives (TN): 957   |  False Positives (FP): 16 ]
  [ False Negatives (FN): 29   |  True Positives  (TP): 0  ]

Integer Underflow / Overflow — With Values:
  [ True Negatives (TN): 881   |  False Positives (FP): 82 ]
  [ False Negatives (FN): 128  |  True Positives  (TP): 75 ]

Integer Underflow / Overflow — Without Values:
  [ True Negatives (TN): 888   |  False Positives (FP): 75 ]
  [ False Negatives (FN): 122  |  True Positives  (TP): 81 ]

Reentrancy — With Values:
  [ True Negatives (TN): 916   |  False Positives (FP): 51 ]
  [ False Negatives (FN): 43   |  True Positives  (TP): 27 ]

Reentrancy — Without Values:
  [ True Negatives (TN): 938   |  False Positives (FP): 29 ]
  [ False Negatives (FN): 48   |  True Positives  (TP): 22 ]
```

---

## 11. Metric Comparisons & Visualizations

The generated visualization charts have been saved into `results/`:
- **F1-Score Comparison**: `results/f1_comparison.png`
- **Accuracy Comparison**: `results/accuracy_comparison.png`
- **Precision Comparison**: `results/precision_comparison.png`
- **Recall Comparison**: `results/recall_comparison.png`
- **Confusion Matrices Grid**: `results/confusion_matrices.png`

---

## 12. Key Observations
1. **Vocabulary Reduction & Computational Efficiency**:
   Across all four datasets, removing opcode values reduced the TF-IDF feature vocabulary size by **37% to 41%** (from ~65,500 features down to ~38,000-41,000 features). This led to a **1.9x to 2.8x speedup** in model training time.
2. **Integer Underflow / Overflow**:
   Removing opcode values improved performance across **all** metrics: Accuracy (+1.11 pp), Precision (+4.15 pp), Recall (+2.96 pp), F1 (+3.46 pp), and ROC-AUC (+0.0184). Stripping contract-specific addresses/constants reduced noise and improved generalization for integer arithmetic patterns.
3. **Etherlock**:
   Removing opcode values slightly decreased F1 (-3.64 pp) and Recall (-5.97 pp), indicating that specific address or length constants in Etherlock bytecodes carry discriminative signal for contract lock detection.
4. **Reentrancy**:
   Precision increased significantly (+8.52 pp from 34.62% to 43.14%), while Recall dropped (-7.14 pp), leaving F1 nearly unchanged (-0.12 pp). Stripping operand values reduced false positives from 51 down to 29.
5. **Block Dependency**:
   Both configurations struggled due to severe class imbalance (only 114 positive samples out of 5,006), yielding 0 True Positives in both baseline and ablation.

---

## 13. Conclusion
Removing opcode operand values during EVM bytecode preprocessing is a double-edged sword:
- It **consistently reduces feature dimensionality (~40%)** and **more than doubles training speed (2x-2.8x faster)**.
- For vulnerability types tied to control flow structures (such as **Integer Underflow/Overflow**), removing contract-specific operand noise **improves predictive performance**.
- For vulnerabilities where specific constant values matter (such as **Etherlock**), stripping operand values causes a mild drop in recall.

---

## 14. Limitations
- **Model Choice**: The study evaluated an unpruned single Decision Tree model.
- **Dataset Imbalance**: Extreme class imbalance in datasets like Block Dependency (2.2% positive class) suppresses precision and recall.
- **Static N-grams**: TF-IDF n-grams (1, 2) capture local sequences but cannot capture long-range control-flow dependencies.

---

## 15. Exact Files and Functions Modified / Added
- [`backend/preprocessing.py`](file:///d:/Solidity-Vuln-Detection/backend/preprocessing.py): Implemented `preprocess_bytecode(bytecode, use_opcode_values, use_mnemonics)` for EVM opcode parsing and operand stripping.
- [`backend/experiment.py`](file:///d:/Solidity-Vuln-Detection/backend/experiment.py): Created unified experiment runner `run_full_ablation_study()`, metric calculators, validity checks, and plot generators.
- [`backend/main.py`](file:///d:/Solidity-Vuln-Detection/backend/main.py): Updated backend configuration options.
- [`results/`](file:///d:/Solidity-Vuln-Detection/results/): Output directory storing `metrics_comparison.csv`, `experiment_summary.json`, and 5 high-resolution plot images.
