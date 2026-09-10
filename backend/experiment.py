import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_bytecode

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "Etherlock",
        "path": "dataset/etherlock.csv",
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    },
    {
        "name": "Block Dependency",
        "path": "dataset/block_dependancy.csv",
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    },
    {
        "name": "Integer Underflow / Overflow",
        "path": "dataset/integer.csv",
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    },
    {
        "name": "Reentrancy",
        "path": "dataset/reentrancy.csv",
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    }
]

def run_single_experiment(dataset_cfg, use_opcode_values: bool, seed: int = 42):
    """
    Executes a single model pipeline run on a dataset.
    """
    df = pd.read_csv(dataset_cfg["path"])
    df = df.dropna(subset=[dataset_cfg["bytecode_col"], dataset_cfg["label_col"]])
    
    raw_bytecodes = df[dataset_cfg["bytecode_col"]].values
    y = df[dataset_cfg["label_col"]].values.astype(int)
    
    # Preprocess corpus
    t0_prep = time.time()
    corpus = [preprocess_bytecode(b, use_opcode_values=use_opcode_values) for b in raw_bytecodes]
    prep_time = time.time() - t0_prep
    
    # Train / Test split (80/20, fixed seed=42)
    indices = np.arange(len(y))
    corpus_train, corpus_test, y_train, y_test, idx_train, idx_test = train_test_split(
        corpus, y, indices, test_size=0.2, random_state=seed
    )
    
    # Vectorizer
    vectorizer = TfidfVectorizer(
        analyzer='word', token_pattern=r'\b\w+\b', ngram_range=(1, 2), min_df=1
    )
    
    # Training
    t0_train = time.time()
    X_train = vectorizer.fit_transform(corpus_train)
    model = DecisionTreeClassifier(
        criterion='gini', max_depth=None, random_state=seed, min_samples_split=2, min_samples_leaf=1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0_train
    
    # Inference
    t0_inf = time.time()
    X_test = vectorizer.transform(corpus_test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    inference_time = time.time() - t0_inf
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
    
    return {
        "dataset": dataset_cfg["name"],
        "use_opcode_values": use_opcode_values,
        "config_name": "With Opcode Values" if use_opcode_values else "Without Opcode Values",
        "num_rows": len(df),
        "vocab_size": X_train.shape[1],
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm.tolist(),
        "prep_time": prep_time,
        "train_time": train_time,
        "inference_time": inference_time,
        "idx_train": idx_train.tolist(),
        "idx_test": idx_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }


def run_full_ablation_study():
    print("=================================================================", flush=True)
    print("STARTING SMART-CONTRACT OPCODE PREPROCESSING ABLATION STUDY", flush=True)
    print("=================================================================", flush=True)
    
    all_results = []
    comparison_table = []
    
    for dcfg in DATASETS:
        dname = dcfg["name"]
        print(f"\n>>> Running Experiments for Dataset: {dname}", flush=True)
        
        # 1. Baseline Run (With Opcode Values)
        print("  - Running Baseline (WITH Opcode Values)...", flush=True)
        res_with = run_single_experiment(dcfg, use_opcode_values=True, seed=42)
        
        # 2. Ablation Run (Without Opcode Values)
        print("  - Running Ablation (WITHOUT Opcode Values)...", flush=True)
        res_without = run_single_experiment(dcfg, use_opcode_values=False, seed=42)
        
        # 3. Validity Check
        assert res_with["num_rows"] == res_without["num_rows"], "Row count mismatch!"
        assert res_with["idx_train"] == res_without["idx_train"], "Train split mismatch!"
        assert res_with["idx_test"] == res_without["idx_test"], "Test split mismatch!"
        assert res_with["y_test"] == res_without["y_test"], "Target label mismatch!"
        print("  [OK] Phase 7 Validity Check PASSED: Identical rows, labels, seed, & splits.", flush=True)
        
        all_results.append(res_with)
        all_results.append(res_without)
        
        # Calculate metric changes
        d_acc = res_without["accuracy"] - res_with["accuracy"]
        d_prec = res_without["precision"] - res_with["precision"]
        d_rec = res_without["recall"] - res_with["recall"]
        d_f1 = res_without["f1"] - res_with["f1"]
        d_roc = res_without["roc_auc"] - res_with["roc_auc"]
        
        comparison_table.append({
            "Dataset": dname,
            "Baseline Acc": res_with["accuracy"],
            "Ablation Acc": res_without["accuracy"],
            "Delta Acc (pp)": d_acc * 100,
            
            "Baseline Prec": res_with["precision"],
            "Ablation Prec": res_without["precision"],
            "Delta Prec (pp)": d_prec * 100,
            
            "Baseline Rec": res_with["recall"],
            "Ablation Rec": res_without["recall"],
            "Delta Rec (pp)": d_rec * 100,
            
            "Baseline F1": res_with["f1"],
            "Ablation F1": res_without["f1"],
            "Delta F1 (pp)": d_f1 * 100,
            
            "Baseline ROC-AUC": res_with["roc_auc"],
            "Ablation ROC-AUC": res_without["roc_auc"],
            "Delta ROC-AUC (pp)": d_roc * 100,
            
            "Vocab With": res_with["vocab_size"],
            "Vocab Without": res_without["vocab_size"],
            "Train Time With (s)": res_with["train_time"],
            "Train Time Without (s)": res_without["train_time"]
        })
    
    # Save JSON summary
    json_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw experimental results to: {json_path}", flush=True)
    
    # Save CSV comparison table
    df_comp = pd.DataFrame(comparison_table)
    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"Saved metrics comparison table to: {csv_path}", flush=True)
    
    # Render Plots
    generate_visualization_plots(all_results)
    
    print("\n=================================================================", flush=True)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)
    print("=================================================================", flush=True)
    return df_comp, all_results


def generate_visualization_plots(results):
    print("\nGenerating Phase 6 visualization plots...", flush=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    # Convert results list to DataFrame
    rows = []
    for r in results:
        rows.append({
            "Dataset": r["dataset"],
            "Configuration": r["config_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "ROC-AUC": r["roc_auc"]
        })
    df_plot = pd.DataFrame(rows)
    
    metrics = ["F1-Score", "Accuracy", "Precision", "Recall"]
    filenames = ["f1_comparison.png", "accuracy_comparison.png", "precision_comparison.png", "recall_comparison.png"]
    colors = ["#2b5c8f", "#d95f02"]
    
    for metric, fname in zip(metrics, filenames):
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(
            data=df_plot,
            x="Dataset",
            y=metric,
            hue="Configuration",
            palette=colors,
            ax=ax,
            edgecolor="black",
            linewidth=1.2
        )
        
        ax.set_title(f"{metric} Comparison: With Opcode Values vs Without Opcode Values", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12, fontweight='bold')
        ax.legend(title="Preprocessing Configuration", frameon=True, facecolor='white', framealpha=0.9)
        
        # Annotate bars with values
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points'
                )
                
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, fname)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  [OK] Saved plot: {plot_path}", flush=True)

    # 5. Plot Confusion Matrices Grid (4 datasets x 2 configs)
    fig, axes = plt.subplots(4, 2, figsize=(11, 18))
    fig.suptitle("Confusion Matrices: With Opcode Values vs Without Opcode Values", fontsize=16, fontweight='bold', y=0.995)
    
    datasets = df_plot["Dataset"].unique()
    
    for i, dname in enumerate(datasets):
        # Baseline (With)
        r_with = [r for r in results if r["dataset"] == dname and r["use_opcode_values"]][0]
        cm_with = np.array(r_with["cm"])
        
        # Ablation (Without)
        r_without = [r for r in results if r["dataset"] == dname and not r["use_opcode_values"]][0]
        cm_without = np.array(r_without["cm"])
        
        # Plot Baseline
        ax_w = axes[i, 0]
        sns.heatmap(cm_with, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_w, annot_kws={"size": 13, "weight": "bold"})
        ax_w.set_title(f"{dname} - With Values", fontsize=12, fontweight='bold')
        ax_w.set_xlabel("Predicted Label", fontsize=10)
        ax_w.set_ylabel("True Label", fontsize=10)
        
        # Plot Ablation
        ax_wo = axes[i, 1]
        sns.heatmap(cm_without, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax_wo, annot_kws={"size": 13, "weight": "bold"})
        ax_wo.set_title(f"{dname} - Without Values", fontsize=12, fontweight='bold')
        ax_wo.set_xlabel("Predicted Label", fontsize=10)
        ax_wo.set_ylabel("True Label", fontsize=10)
        
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved confusion matrices plot: {cm_path}", flush=True)


if __name__ == "__main__":
    run_full_ablation_study()
