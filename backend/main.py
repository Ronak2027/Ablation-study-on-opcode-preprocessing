import os
import sys
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from preprocessing import preprocess_bytecode

try:
    import solcx
    from solcx import compile_source, install_solc
    try:
        solcx.set_solc_version("0.8.17")
    except Exception:
        try:
            install_solc("0.8.17")
            solcx.set_solc_version("0.8.17")
        except Exception as solc_err:
            print(f"[WARN] solc setup notice: {solc_err}")
except ImportError:
    solcx = None

app = FastAPI(
    title="Smart Contract Opcode Preprocessing & Vulnerability Detection API",
    description="Ablation-Study-Ready Detection API supporting both baseline and ablated opcode preprocessing pipelines.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Vulnerability metadata
VULN_CONFIG = [
    {"name": "Etherlock", "key": "etherlock"},
    {"name": "Block Dependency", "key": "block_dependancy"},
    {"name": "Underflow / Overflow", "key": "integer"},
    {"name": "Reentrancy", "key": "reentrancy"}
]

# Pipeline registry loading fitted models and vectorizers
# Stores both baseline ("with_operands") and ablation ("without_operands")
PIPELINES = {
    "with_operands": {},
    "without_operands": {}
}

MODELS_DIR = os.path.join(BASE_DIR, "models")
MANIFEST_PATH = os.path.join(BASE_DIR, "model_manifest.json")


def load_pipelines():
    """
    Loads saved DecisionTree models alongside their fitted TfidfVectorizer instances.
    Supports both baseline ('with_operands') and ablation ('without_operands') configurations.
    """
    for item in VULN_CONFIG:
        v_name = item["name"]
        v_key = item["key"]

        # 1. Baseline: With Operands
        model_path_with = os.path.join(MODELS_DIR, f"{v_key}_with_operands_model.joblib")
        vec_path_with = os.path.join(MODELS_DIR, f"{v_key}_with_operands_vectorizer.joblib")
        # Fallback to legacy files if models/ not used
        if not os.path.exists(model_path_with):
            model_path_with = os.path.join(BASE_DIR, f"{v_key}_decisiontree.joblib")
            vec_path_with = os.path.join(BASE_DIR, f"{v_key}_vectorizer.joblib")

        # 2. Ablation: Without Operands
        model_path_without = os.path.join(MODELS_DIR, f"{v_key}_without_operands_model.joblib")
        vec_path_without = os.path.join(MODELS_DIR, f"{v_key}_without_operands_vectorizer.joblib")

        # Load baseline
        if os.path.exists(model_path_with) and os.path.exists(vec_path_with):
            model_w = joblib.load(model_path_with)
            vec_w = joblib.load(vec_path_with)
            PIPELINES["with_operands"][v_name] = {
                "model": model_w,
                "vectorizer": vec_w,
                "vocab_size": model_w.n_features_in_
            }
        else:
            print(f"[WARN] Missing baseline artifacts for {v_name}")

        # Load ablation
        if os.path.exists(model_path_without) and os.path.exists(vec_path_without):
            model_wo = joblib.load(model_path_without)
            vec_wo = joblib.load(vec_path_without)
            PIPELINES["without_operands"][v_name] = {
                "model": model_wo,
                "vectorizer": vec_wo,
                "vocab_size": model_wo.n_features_in_
            }
        else:
            print(f"[WARN] Missing ablation artifacts for {v_name}")

    print(f"[INFO] Loaded pipelines: {len(PIPELINES['with_operands'])} baseline, {len(PIPELINES['without_operands'])} ablation.")


# Load models and vectorizers on startup
load_pipelines()


# Pydantic Schemas
class ContractItem(BaseModel):
    filename: str
    content: str  # Can be Solidity source code OR raw bytecode hex


class AnalyzeRequest(BaseModel):
    contracts: List[ContractItem]
    # Options: "both" (default, runs both pipelines for ablation comparison),
    # "with_operands" (baseline only),
    # "without_operands" (ablation only)
    mode: Optional[str] = "both"


class PreprocessRequest(BaseModel):
    content: str  # Solidity source or hex bytecode


def extract_bytecode(content: str) -> str:
    """
    Extracts or compiles bytecode from content.
    If content is already hex bytecode (or 0x-prefixed), returns cleaned hex string.
    If content is Solidity source, compiles via solcx.
    """
    clean_str = content.strip()
    
    # Check if raw hex bytecode (e.g. starts with 0x60, 6080, 6060, or pure hex)
    is_hex_prefix = clean_str.startswith("0x") or clean_str.startswith("0X")
    hex_body = clean_str[2:] if is_hex_prefix else clean_str
    
    # Simple heuristic: if it looks like compiled hex and doesn't contain solidity keywords
    if len(hex_body) >= 10 and all(c in "0123456789abcdefABCDEF \n\r\t" for c in hex_body):
        cleaned_hex = "".join(c for c in hex_body if c in "0123456789abcdefABCDEF")
        if len(cleaned_hex) % 2 != 0:
            cleaned_hex = cleaned_hex[:-1]
        return cleaned_hex

    # Otherwise compile with solcx
    if solcx is None:
        raise RuntimeError("Solidity compiler (solcx) is not available.")
    
    try:
        compiled = compile_source(content, output_values=["evm.bytecode.object"])
        if not compiled:
            raise ValueError("Compilation succeeded but returned empty contracts object.")
        first_contract = list(compiled.values())[0]
        bytecode = first_contract["evm"]["bytecode"]["object"]
        if not bytecode:
            raise ValueError("Solidity contract produced empty bytecode (e.g., interface or abstract contract).")
        return bytecode
    except Exception as e:
        raise RuntimeError(f"Solidity compilation failed: {str(e)}")


def run_inference_for_config(bytecode: str, use_opcode_values: bool) -> Dict[str, Any]:
    """
    Executes inference for a specific opcode preprocessing configuration using
    the matched fitted TfidfVectorizer and DecisionTree model.
    """
    cfg_key = "with_operands" if use_opcode_values else "without_operands"
    pipeline_group = PIPELINES.get(cfg_key, {})

    if not pipeline_group:
        raise RuntimeError(f"No pipelines loaded for configuration: {cfg_key}")

    # Use the research experiment's exact opcode-aware preprocessor
    tokenized_str = preprocess_bytecode(bytecode, use_opcode_values=use_opcode_values)
    token_list = tokenized_str.split() if tokenized_str else []
    token_count = len(token_list)

    vulnerabilities = {}
    probabilities = {}

    for vuln_name, pipe in pipeline_group.items():
        vec = pipe["vectorizer"]
        model = pipe["model"]

        # CRITICAL FIX 2: Use vectorizer.transform(), NEVER fit_transform()
        X = vec.transform([tokenized_str])
        pred = int(model.predict(X)[0])
        vulnerabilities[vuln_name] = pred

        if hasattr(model, "predict_proba"):
            try:
                proba = float(model.predict_proba(X)[0][1])
                probabilities[vuln_name] = round(proba, 4)
            except Exception:
                probabilities[vuln_name] = float(pred)
        else:
            probabilities[vuln_name] = float(pred)

    return {
        "configuration": cfg_key,
        "config_label": "With Opcode Values" if use_opcode_values else "Without Opcode Values",
        "token_count": token_count,
        "tokens_preview": " ".join(token_list[:30]) + ("..." if len(token_list) > 30 else ""),
        "vulnerabilities": vulnerabilities,
        "probabilities": probabilities
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "baseline_models_loaded": len(PIPELINES["with_operands"]),
        "ablation_models_loaded": len(PIPELINES["without_operands"]),
        "categories": [item["name"] for item in VULN_CONFIG]
    }


@app.get("/manifest")
def get_manifest():
    """
    Returns the experimental model manifest detailing exact vocabulary sizes,
    feature counts, and model metrics for both configurations.
    """
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model manifest not found. Run experiment.py to generate it.")


@app.get("/metrics")
def get_metrics_table():
    """
    Returns the CSV metrics comparison as structured JSON.
    """
    csv_path = os.path.join(PROJECT_ROOT, "results", "metrics_comparison.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return {"data": df.to_dict(orient="records")}
    raise HTTPException(status_code=404, detail="Metrics comparison table not found.")


@app.post("/preprocess")
def preprocess_preview(request: PreprocessRequest):
    """
    Interactive demonstration of EVM opcode preprocessing:
    returns bytecode, tokenization with operands, and tokenization without operands.
    """
    try:
        bytecode = extract_bytecode(request.content)
        tokens_with = preprocess_bytecode(bytecode, use_opcode_values=True)
        tokens_without = preprocess_bytecode(bytecode, use_opcode_values=False)
        
        count_with = len(tokens_with.split()) if tokens_with else 0
        count_without = len(tokens_without.split()) if tokens_without else 0
        reduction_pct = round(((count_with - count_without) / count_with * 100), 2) if count_with > 0 else 0.0

        return {
            "bytecode_length_chars": len(bytecode),
            "bytecode_preview": bytecode[:100] + ("..." if len(bytecode) > 100 else ""),
            "with_operands": {
                "token_count": count_with,
                "tokens": tokens_with[:300] + ("..." if len(tokens_with) > 300 else "")
            },
            "without_operands": {
                "token_count": count_without,
                "tokens": tokens_without[:300] + ("..." if len(tokens_without) > 300 else "")
            },
            "token_reduction_pct": reduction_pct
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze")
def analyze_contracts(request: AnalyzeRequest):
    """
    Analyzes smart contracts for vulnerabilities.
    Supports both baseline and ablation preprocessing configurations.
    """
    results = []
    mode = (request.mode or "both").lower()

    for contract in request.contracts:
        try:
            bytecode = extract_bytecode(contract.content)
            
            res_with = None
            res_without = None

            if mode in ["both", "with_operands", "baseline"]:
                res_with = run_inference_for_config(bytecode, use_opcode_values=True)

            if mode in ["both", "without_operands", "ablation"]:
                res_without = run_inference_for_config(bytecode, use_opcode_values=False)

            # Determine primary vulnerabilities output for backward compatibility
            primary_vulnerabilities = (
                res_with["vulnerabilities"] if res_with
                else (res_without["vulnerabilities"] if res_without else {})
            )

            contract_entry = {
                "filename": contract.filename,
                "bytecode_length": len(bytecode),
                "vulnerabilities": primary_vulnerabilities,  # backwards compatibility
                "mode": mode,
                "baseline": res_with,
                "ablation": res_without
            }

            if res_with and res_without:
                token_reduction = round(
                    ((res_with["token_count"] - res_without["token_count"]) / res_with["token_count"] * 100), 2
                ) if res_with["token_count"] > 0 else 0.0
                
                # Check agreement across vulnerability detectors
                agreements = {
                    k: (res_with["vulnerabilities"].get(k) == res_without["vulnerabilities"].get(k))
                    for k in res_with["vulnerabilities"]
                }
                contract_entry["comparison"] = {
                    "token_reduction_pct": token_reduction,
                    "agreements": agreements,
                    "all_agree": all(agreements.values())
                }

            results.append(contract_entry)

        except Exception as e:
            results.append({
                "filename": contract.filename,
                "error": str(e),
                "vulnerabilities": {
                    "Etherlock": 0,
                    "Reentrancy": 0,
                    "Block Dependency": 0,
                    "Underflow / Overflow": 0
                },
                "baseline": None,
                "ablation": None
            })

    return {"results": results, "mode": mode}
