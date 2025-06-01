from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import joblib
import solcx
from solcx import compile_source, install_solc
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    solcx.set_solc_version("0.8.17")
except solcx.exceptions.SolcNotInstalled:
    install_solc("0.8.17")
    solcx.set_solc_version("0.8.17")

MODELS = {
    "Etherlock": joblib.load("etherlock_decisiontree.joblib"),
    "Reentrancy": joblib.load("reentrancy_decisiontree.joblib"),
    "Block Dependency": joblib.load("block_dependancy_decisiontree.joblib"),
    "Underflow / Overflow": joblib.load("integer_decisiontree.joblib"),
}
vectorizer = TfidfVectorizer(
    analyzer='word', token_pattern=r'\b\w+\b', ngram_range=(1, 2), min_df=1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class Contract(BaseModel):
    filename: str
    content: str


class AnalyzeRequest(BaseModel):
    contracts: List[Contract]


def compile_contract(source: str) -> str:
    try:
        compiled = compile_source(
            source,
            output_values=["evm.bytecode.object"]
        )
        bytecode = list(compiled.values())[0]["evm"]["bytecode"]["object"]
        return bytecode
    except Exception as e:
        raise RuntimeError(f"Solidity compilation failed: {str(e)}")


def bytecode_to_tokens(bytecode: str) -> str:
    return ' '.join(bytecode[i:i+2] for i in range(0, len(bytecode), 2))


@app.post("/analyze")
def analyze_contracts(request: AnalyzeRequest):
    results = []

    for contract in request.contracts:
        try:
            bytecode = compile_contract(contract.content)
            tokenized = bytecode_to_tokens(bytecode)
            X = vectorizer.fit_transform([tokenized])

            vulnerabilities = {}
            for vuln_name, model in MODELS.items():
                pred = model.predict(X)
                vulnerabilities[vuln_name] = int(pred[0])

            results.append({
                "filename": contract.filename,
                "vulnerabilities": vulnerabilities
            })

        except Exception as e:
            results.append({
                "filename": contract.filename,
                "error": str(e),
                "vulnerabilities": {
                    "Etherlock": 0,
                    "Reentrancy": 0,
                    "Block Dependency": 0,
                    "Underflow / Overflow": 0
                }
            })

    return {"results": results}
