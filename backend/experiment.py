import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "Etherlock",
        "key": "etherlock",
        "path": os.path.join(PROJECT_ROOT, "dataset", "etherlock.csv"),
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    },
    {
        "name": "Block Dependency",
        "key": "block_dependancy",
        "path": os.path.join(PROJECT_ROOT, "dataset", "block_dependancy.csv"),
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    },
    {
        "name": "Integer Underflow / Overflow",
        "key": "integer",
        "path": os.path.join(PROJECT_ROOT, "dataset", "integer.csv"),
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    },
    {
        "name": "Reentrancy",
        "key": "reentrancy",
        "path": os.path.join(PROJECT_ROOT, "dataset", "reentrancy.csv"),
        "label_col": "is_vulnerable",
        "bytecode_col": "bytecode"
    }
]

def run_single_experiment(dataset_cfg, use_opcode_values: bool, seed: int = 42):
    """
    Executes a single model pipeline run on a dataset.
    Isolates TF-IDF vectorizer fitting time from Decision Tree classifier fitting time.
    """
    df = pd.read_csv(dataset_cfg["path"])
    df = df.dropna(subset=[dataset_cfg["bytecode_col"], dataset_cfg["label_col"]])
    
    raw_bytecodes = df[dataset_cfg["bytecode_col"]].values
    y = df[dataset_cfg["label_col"]].values.astype(int)
    
    # Preprocess corpus using opcode-aware parser
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
    
    # 1. Measure TF-IDF vectorizer fitting time separately
    t0_tfidf = time.time()
    X_train = vectorizer.fit_transform(corpus_train)
    tfidf_fit_time = time.time() - t0_tfidf
    
    # 2. Measure Decision Tree classifier fitting time separately
    model = DecisionTreeClassifier(
        criterion='gini', max_depth=None, random_state=seed, min_samples_split=2, min_samples_leaf=1
    )
    t0_clf = time.time()
    model.fit(X_train, y_train)
    clf_fit_time = time.time() - t0_clf
    
    # Combined pipeline training time (TF-IDF fitting + classifier fitting)
    total_train_time = tfidf_fit_time + clf_fit_time
    
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
    
    vocab_size = int(X_train.shape[1])
    assert model.n_features_in_ == vocab_size == len(vectorizer.vocabulary_), (
        f"Feature mismatch: model expects {model.n_features_in_}, "
        f"matrix has {vocab_size}, vectorizer has {len(vectorizer.vocabulary_)}"
    )
    
    return {
        "dataset": dataset_cfg["name"],
        "dataset_key": dataset_cfg["key"],
        "use_opcode_values": use_opcode_values,
        "config_name": "With Opcode Values" if use_opcode_values else "Without Opcode Values",
        "num_rows": int(len(df)),
        "vocab_size": vocab_size,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "cm": cm.tolist(),
        "prep_time": float(prep_time),
        "tfidf_fit_time": float(tfidf_fit_time),
        "clf_fit_time": float(clf_fit_time),
        "total_train_time": float(total_train_time),
        "train_time": float(total_train_time),  # legacy compatibility
        "inference_time": float(inference_time),
        "idx_train": idx_train.tolist(),
        "idx_test": idx_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "_model": model,
        "_vectorizer": vectorizer
    }


def save_model_artifacts(res):
    """
    Saves trained DecisionTreeClassifier and fitted TfidfVectorizer together.
    Also creates legacy files for backwards compatibility.
    """
    key = res["dataset_key"]
    use_opcode_values = res["use_opcode_values"]
    cfg_tag = "with_operands" if use_opcode_values else "without_operands"
    
    # Save into backend/models/
    model_path = os.path.join(MODELS_DIR, f"{key}_{cfg_tag}_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, f"{key}_{cfg_tag}_vectorizer.joblib")
    
    joblib.dump(res["_model"], model_path)
    joblib.dump(res["_vectorizer"], vectorizer_path)
    
    legacy_model_path = None
    legacy_vec_path = None
    # For baseline default ("with_operands"), also update legacy backend/*.joblib files
    if use_opcode_values:
        legacy_model_path = os.path.join(BASE_DIR, f"{key}_decisiontree.joblib")
        legacy_vec_path = os.path.join(BASE_DIR, f"{key}_vectorizer.joblib")
        joblib.dump(res["_model"], legacy_model_path)
        joblib.dump(res["_vectorizer"], legacy_vec_path)
    
    return {
        "model_file": os.path.relpath(model_path, PROJECT_ROOT).replace("\\", "/"),
        "vectorizer_file": os.path.relpath(vectorizer_path, PROJECT_ROOT).replace("\\", "/"),
        "legacy_model_file": os.path.relpath(legacy_model_path, PROJECT_ROOT).replace("\\", "/") if legacy_model_path else None,
        "legacy_vectorizer_file": os.path.relpath(legacy_vec_path, PROJECT_ROOT).replace("\\", "/") if legacy_vec_path else None,
    }


def run_full_ablation_study():
    print("=================================================================", flush=True)
    print("STARTING SMART-CONTRACT OPCODE PREPROCESSING ABLATION STUDY", flush=True)
    print("=================================================================", flush=True)
    
    all_results = []
    manifest_entries = []
    comparison_table = []
    
    for dcfg in DATASETS:
        dname = dcfg["name"]
        print(f"\n>>> Running Experiments for Dataset: {dname}", flush=True)
        
        # 1. Baseline Run (With Opcode Values)
        print("  - Running Baseline (WITH Opcode Values)...", flush=True)
        res_with = run_single_experiment(dcfg, use_opcode_values=True, seed=42)
        paths_with = save_model_artifacts(res_with)
        
        # 2. Ablation Run (Without Opcode Values)
        print("  - Running Ablation (WITHOUT Opcode Values)...", flush=True)
        res_without = run_single_experiment(dcfg, use_opcode_values=False, seed=42)
        paths_without = save_model_artifacts(res_without)
        
        # 3. Validity Check
        assert res_with["num_rows"] == res_without["num_rows"], "Row count mismatch!"
        assert res_with["idx_train"] == res_without["idx_train"], "Train split mismatch!"
        assert res_with["idx_test"] == res_without["idx_test"], "Test split mismatch!"
        assert res_with["y_test"] == res_without["y_test"], "Target label mismatch!"
        print("  [OK] Validity Check PASSED: Identical rows, labels, seed, & splits.", flush=True)
        print(f"  [OK] Baseline Vocab: {res_with['vocab_size']:,} | Ablation Vocab: {res_without['vocab_size']:,}", flush=True)
        print(f"  [OK] TF-IDF Time: {res_with['tfidf_fit_time']:.2f}s vs {res_without['tfidf_fit_time']:.2f}s", flush=True)
        print(f"  [OK] Classifier Time: {res_with['clf_fit_time']:.2f}s vs {res_without['clf_fit_time']:.2f}s", flush=True)
        print(f"  [OK] Total Fit Time: {res_with['total_train_time']:.2f}s vs {res_without['total_train_time']:.2f}s", flush=True)
        
        # Create manifest entries
        for r, p in [(res_with, paths_with), (res_without, paths_without)]:
            manifest_entries.append({
                "dataset_name": r["dataset"],
                "dataset_key": r["dataset_key"],
                "configuration": "with_operands" if r["use_opcode_values"] else "without_operands",
                "config_label": r["config_name"],
                "model_file": p["model_file"],
                "vectorizer_file": p["vectorizer_file"],
                "legacy_model_file": p["legacy_model_file"],
                "legacy_vectorizer_file": p["legacy_vectorizer_file"],
                "vocab_size": r["vocab_size"],
                "expected_feature_count": r["vocab_size"],
                "metrics": {
                    "accuracy": r["accuracy"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "roc_auc": r["roc_auc"],
                    "confusion_matrix": r["cm"]
                },
                "timings_seconds": {
                    "preprocessing_time": r["prep_time"],
                    "tfidf_fit_time": r["tfidf_fit_time"],
                    "classifier_fit_time": r["clf_fit_time"],
                    "total_pipeline_train_time": r["total_train_time"],
                    "inference_time": r["inference_time"]
                }
            })
        
        # Clean results for JSON summary (strip internal model objects)
        clean_with = {k: v for k, v in res_with.items() if not k.startswith("_")}
        clean_without = {k: v for k, v in res_without.items() if not k.startswith("_")}
        all_results.append(clean_with)
        all_results.append(clean_without)
        
        # Calculate metric changes
        d_acc = res_without["accuracy"] - res_with["accuracy"]
        d_prec = res_without["precision"] - res_with["precision"]
        d_rec = res_without["recall"] - res_with["recall"]
        d_f1 = res_without["f1"] - res_with["f1"]
        d_roc = res_without["roc_auc"] - res_with["roc_auc"]
        d_vocab_pct = ((res_without["vocab_size"] - res_with["vocab_size"]) / res_with["vocab_size"]) * 100
        
        tfidf_speedup = res_with["tfidf_fit_time"] / res_without["tfidf_fit_time"] if res_without["tfidf_fit_time"] > 0 else 1.0
        clf_speedup = res_with["clf_fit_time"] / res_without["clf_fit_time"] if res_without["clf_fit_time"] > 0 else 1.0
        total_speedup = res_with["total_train_time"] / res_without["total_train_time"] if res_without["total_train_time"] > 0 else 1.0
        
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
            "Delta Vocab (%)": d_vocab_pct,
            
            # Isolated TF-IDF Fitting Times
            "Baseline TF-IDF Fit Time (s)": res_with["tfidf_fit_time"],
            "Ablation TF-IDF Fit Time (s)": res_without["tfidf_fit_time"],
            "TF-IDF Speedup (x)": tfidf_speedup,
            
            # Isolated Decision Tree Classifier Fitting Times
            "Baseline Classifier Fit Time (s)": res_with["clf_fit_time"],
            "Ablation Classifier Fit Time (s)": res_without["clf_fit_time"],
            "Classifier Speedup (x)": clf_speedup,
            
            # Combined Pipeline Fitting Times (TF-IDF + Classifier)
            "Baseline Total Fit Time (s)": res_with["total_train_time"],
            "Ablation Total Fit Time (s)": res_without["total_train_time"],
            "Total Fit Speedup (x)": total_speedup,
            
            # Backward compatibility fields
            "Train Time With (s)": res_with["total_train_time"],
            "Train Time Without (s)": res_without["total_train_time"]
        })
    
    # Save Model Manifest to both backend/ and results/
    manifest_data = {
        "title": "Smart Contract Opcode Preprocessing Ablation Model Manifest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scikit_learn_version": joblib.__version__,
        "models": manifest_entries
    }
    
    backend_manifest_path = os.path.join(BASE_DIR, "model_manifest.json")
    with open(backend_manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\n[OK] Saved Model Manifest to: {backend_manifest_path}", flush=True)
    
    results_manifest_path = os.path.join(RESULTS_DIR, "model_manifest.json")
    with open(results_manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[OK] Saved Model Manifest copy to: {results_manifest_path}", flush=True)
    
    # Save JSON summary
    json_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Saved raw experimental results to: {json_path}", flush=True)
    
    # Save CSV comparison table
    df_comp = pd.DataFrame(comparison_table)
    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"[OK] Saved metrics comparison table to: {csv_path}", flush=True)
    
    # Render Plots
    generate_visualization_plots(all_results, comparison_table)
    
    print("\n=================================================================", flush=True)
    print("ALL EXPERIMENTS & ARTIFACTS GENERATED SUCCESSFULLY!", flush=True)
    print("=================================================================", flush=True)
    return df_comp, all_results


def generate_visualization_plots(results, comparison_table):
    print("\nGenerating visualization plots...", flush=True)
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
            "ROC-AUC": r["roc_auc"],
            "TF-IDF Fit Time (s)": r["tfidf_fit_time"],
            "Classifier Fit Time (s)": r["clf_fit_time"],
            "Total Pipeline Fit Time (s)": r["total_train_time"]
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
        r_with = [r for r in results if r["dataset"] == dname and r["use_opcode_values"]][0]
        cm_with = np.array(r_with["cm"])
        
        r_without = [r for r in results if r["dataset"] == dname and not r["use_opcode_values"]][0]
        cm_without = np.array(r_without["cm"])
        
        ax_w = axes[i, 0]
        sns.heatmap(cm_with, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_w, annot_kws={"size": 13, "weight": "bold"})
        ax_w.set_title(f"{dname} - With Values", fontsize=12, fontweight='bold')
        ax_w.set_xlabel("Predicted Label", fontsize=10)
        ax_w.set_ylabel("True Label", fontsize=10)
        
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

    # 6. Training Time Breakdown Plot (Critical 4 Fix)
    # Visualizes isolated TF-IDF fit time, Decision Tree fit time, and Total pipeline fit time
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Training Time Breakdown: TF-IDF Fitting vs Decision Tree Fitting vs Total Pipeline", fontsize=15, fontweight='bold')
    
    timing_titles = [
        ("TF-IDF Vectorizer Fit Time (s)", "TF-IDF Fit Time (s)"),
        ("Classifier (Decision Tree) Fit Time (s)", "Classifier Fit Time (s)"),
        ("Total Pipeline Training Time (s)", "Total Pipeline Fit Time (s)")
    ]
    
    for ax, (title, col) in zip(axes, timing_titles):
        sns.barplot(
            data=df_plot,
            x="Dataset",
            y=col,
            hue="Configuration",
            palette=colors,
            ax=ax,
            edgecolor="black",
            linewidth=1.2
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel("Time (seconds)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(title="Configuration", frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}s",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points'
                )
    
    plt.tight_layout()
    time_path = os.path.join(RESULTS_DIR, "training_time_breakdown.png")
    plt.savefig(time_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved training time breakdown plot: {time_path}", flush=True)


if __name__ == "__main__":
    run_full_ablation_study()
