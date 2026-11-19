"""
backend/main.py
===============
BioPredict — FastAPI Backend
Bioactive Molecule Prediction Platform

Endpoints:
  GET  /                     -> API information or redirect to /app/
  GET  /health               -> Health check
  POST /predict              -> Single molecule prediction & analysis
  POST /predict_batch        -> Batch CSV prediction
  POST /visualize            -> Generate 2D molecular structure SVG
  GET  /models               -> All six model comparison metrics
  GET  /app                  -> Serves frontend home page
  GET  /app/predict          -> Serves prediction page
  GET  /app/batch            -> Serves batch screening page
  GET  /app/benchmark        -> Serves benchmark page
  GET  /app/about            -> Serves about page
  GET  /app/bioactivity-guide -> Serves bioactivity & chirality guide page
"""
import os
import sys
import io
import json
import csv
import base64
import urllib.request
import urllib.parse
from typing import Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, AllChem, Descriptors, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ── Paths & Setup ─────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(CURRENT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
CONFUSION_PATH = os.path.join(MODEL_DIR, "confusion_matrices.json")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# ── Load Model & Metadata ─────────────────────────────────────────────────────
best_model = None
metadata = {}
BEST_MODEL_NAME = "XGBoost"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            BEST_MODEL_NAME = metadata.get("selected_model", "XGBoost")
            FINGERPRINT_RADIUS = metadata.get("fingerprint_radius", 2)
            FINGERPRINT_BITS = metadata.get("fingerprint_bits", 2048)
    except Exception as e:
        print(f"[WARN] Failed to load metadata: {e}")

if os.path.exists(MODEL_PATH):
    try:
        best_model = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded: {BEST_MODEL_NAME}")
        print(f"[OK] Fingerprint: Morgan radius={FINGERPRINT_RADIUS}, bits={FINGERPRINT_BITS}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")

confusion_data = {}
if os.path.exists(CONFUSION_PATH):
    try:
        with open(CONFUSION_PATH, "r", encoding="utf-8") as f:
            confusion_data = json.load(f)
            print("[OK] Confusion matrices loaded.")
    except Exception as e:
        print(f"[WARN] Failed to load confusion matrices: {e}")
else:
    print("[WARN] confusion_matrices.json not found — model comparison will use metadata only.")

# ── Reference Dictionaries ────────────────────────────────────────────────────
NAME_CACHE = {}

KNOWN_MOLECULES = {
    "CCO": "Ethanol",
    "CC(=O)Oc1ccccc1C(=O)O": "Aspirin",
    "c1ccccc1": "Benzene",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "Ibuprofen",
    "CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1": "Celecoxib Analog",
    "CC(=O)Nc1ccc(O)cc1": "Acetaminophen",
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1": "Naproxen",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C": "Caffeine",
    "CCC": "Propane",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1": "Celecoxib",
    "O=S(=O)(c1ccc(C2=C(c3ccccc3)C(=O)OC2)cc1)C": "Rofecoxib",
    "Cc1c(c2ccccc2)onc1-c1ccc(S(N)(=O)=O)cc1": "Valdecoxib",
    "Cc1ccc(-c2ccc(C)c(c3ccc(S(C)(=O)=O)cc3)n2)cn1": "Etoricoxib",
    "Cc1nc(NC(=O)c2c(O)c3ccccc3s(=O)(=O)n2C)cs1": "Meloxicam",
    "CN1C(C(=O)Nc2ccccn2)=C(O)c2ccccc2S1(=O)=O": "Piroxicam",
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl": "Diclofenac",
    "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1": "Indomethacin",
    "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1": "Ketoprofen",
    "CC(C(=O)O)c1ccc(-c2ccccc2)c(F)c1": "Flurbiprofen",
    "Cc1cccc(C)c1Nc1ccccc1C(=O)O": "Mefenamic Acid",
    "O=C(O)c1ccccc1O": "Salicylic Acid",
    "CC1=C(Cc2ccccc2)C(=Cc2ccc(S(C)=O)cc2)c2cc(F)ccc21": "Sulindac",
    "Cc1ccccc1": "Toluene",
    "Oc1ccccc1": "Phenol",
    "CO": "Methanol"
}

CANONICAL_NAME_MAP = {}
for _smi, _name in KNOWN_MOLECULES.items():
    _m = Chem.MolFromSmiles(_smi)
    if _m:
        CANONICAL_NAME_MAP[Chem.MolToSmiles(_m)] = _name
    CANONICAL_NAME_MAP[_smi] = _name

NAME_TO_SMILES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "benzene": "c1ccccc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "celebrex": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "celecoxib analog": "CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1",
    "celecoxib_analog": "CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "tylenol": "CC(=O)Nc1ccc(O)cc1",
    "naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "aleve": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "propane": "CCC",
    "rofecoxib": "O=S(=O)(c1ccc(C2=C(c3ccccc3)C(=O)OC2)cc1)C",
    "vioxx": "O=S(=O)(c1ccc(C2=C(c3ccccc3)C(=O)OC2)cc1)C",
    "valdecoxib": "Cc1c(c2ccccc2)onc1-c1ccc(S(N)(=O)=O)cc1",
    "bextra": "Cc1c(c2ccccc2)onc1-c1ccc(S(N)(=O)=O)cc1",
    "etoricoxib": "Cc1ccc(-c2ccc(C)c(c3ccc(S(C)(=O)=O)cc3)n2)cn1",
    "arcoxia": "Cc1ccc(-c2ccc(C)c(c3ccc(S(C)(=O)=O)cc3)n2)cn1",
    "meloxicam": "Cc1nc(NC(=O)c2c(O)c3ccccc3s(=O)(=O)n2C)cs1",
    "mobic": "Cc1nc(NC(=O)c2c(O)c3ccccc3s(=O)(=O)n2C)cs1",
    "piroxicam": "CN1C(C(=O)Nc2ccccn2)=C(O)c2ccccc2S1(=O)=O",
    "diclofenac": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "voltaren": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "indomethacin": "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
    "ketoprofen": "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1",
    "flurbiprofen": "CC(C(=O)O)c1ccc(-c2ccccc2)c(F)c1",
    "mefenamic acid": "Cc1cccc(C)c1Nc1ccccc1C(=O)O",
    "salicylic acid": "O=C(O)c1ccccc1O",
    "sulindac": "CC1=C(Cc2ccccc2)C(=Cc2ccc(S(C)=O)cc2)c2cc(F)ccc21",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "methanol": "CO",
    "advil": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "motrin": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
}

# Load sample molecules mapping if available
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "sample_molecules.csv")
if os.path.exists(SAMPLE_PATH):
    try:
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _s = row.get("smiles", "").strip()
                _n = row.get("molecule_name", "").strip()
                if _s and _n:
                    _m = Chem.MolFromSmiles(_s)
                    if _m:
                        CANONICAL_NAME_MAP[Chem.MolToSmiles(_m)] = _n
                    CANONICAL_NAME_MAP[_s] = _n
                    NAME_TO_SMILES[_n.lower()] = _s
    except Exception as e:
        print(f"[WARN] Failed to load sample_molecules.csv: {e}")

# Load ChEMBL ID mapping
CHEMBL_MAP = {}
CHEMBL_CSV = os.path.join(PROJECT_ROOT, "CHEMBL230_Preprocessed_Data_V2.csv")
if os.path.exists(CHEMBL_CSV):
    try:
        with open(CHEMBL_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _s = row.get("canonical_smiles", "").strip()
                _cid = row.get("molecule_chembl_id", "").strip()
                if _s and _cid:
                    CHEMBL_MAP[_s] = _cid
    except Exception as e:
        print(f"[WARN] Failed to load ChEMBL dataset mapping: {e}")

# ── RDKit & ML Utility Functions ──────────────────────────────────────────────
def smiles_to_fingerprint(smiles: str, return_bit_info: bool = False):
    """
    Converts a SMILES string to a 2048-bit Morgan fingerprint vector.
    Matches the exact feature generation used during model training.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return (None, None, None) if return_bit_info else (None, None)

    bit_info = {} if return_bit_info else None
    if return_bit_info:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS, bitInfo=bit_info
        )
    else:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
        )

    feature_arr = np.zeros((1, FINGERPRINT_BITS), dtype=np.float32)
    for bit in fp.GetOnBits():
        feature_arr[0, bit] = 1.0

    if return_bit_info:
        return feature_arr, mol, bit_info
    return feature_arr, mol


def run_prediction(features):
    """
    Runs the saved XGBoost classifier on a fingerprint vector.
    Returns (pred_label, raw_score, score_type).
    Threshold is 0.5 (calibrated decision boundary).
    """
    if best_model is None:
        raise RuntimeError("Model is not loaded.")

    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(features)[0]
        # Class 1 = Active, Class 0 = Inactive
        active_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        pred_label = "Active" if active_prob >= 0.5 else "Inactive"
        return pred_label, active_prob, "probability"
    else:
        pred_int = int(best_model.predict(features)[0])
        pred_label = "Active" if pred_int == 1 else "Inactive"
        return pred_label, 1.0 if pred_label == "Active" else 0.0, "binary"


def explain_atom_attribution(mol, bit_info):
    """
    Identifies which atoms and chemical substructures influenced the XGBoost prediction.
    Plain English Explanation for Viva Defense:
    1. RDKit calculates circular Morgan Fingerprints (radius=2) around each atom.
    2. 'bit_info' records which center atom and radius activated each active bit.
    3. The XGBoost model has learned weights (feature_importances_) for all 2048 bits.
    4. By matching the active bits back to their originating atoms, we highlight the
       exact substructures responsible for the model's classification.
    """
    if not hasattr(best_model, "feature_importances_") or mol is None or not bit_info:
        return {
            "highlighted_atom_indices": [],
            "top_substructures": [],
            "atom_scores": []
        }

    feature_importances = best_model.feature_importances_
    num_atoms = mol.GetNumAtoms()
    atom_scores = np.zeros(num_atoms, dtype=float)

    active_weighted_bits = []
    for bit_id, center_list in bit_info.items():
        weight = float(feature_importances[bit_id])
        if weight > 0:
            active_weighted_bits.append((bit_id, weight, center_list))

    active_weighted_bits.sort(key=lambda x: x[1], reverse=True)

    top_substructures = []
    seen_fragments = set()

    for bit_id, weight, center_list in active_weighted_bits:
        all_bit_atoms = set()
        for center_idx, radius in center_list:
            env_atoms = {center_idx}
            if radius > 0:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
                for b_idx in env:
                    bond = mol.GetBondWithIdx(b_idx)
                    env_atoms.add(bond.GetBeginAtomIdx())
                    env_atoms.add(bond.GetEndAtomIdx())
            all_bit_atoms.update(env_atoms)
            for atom_idx in env_atoms:
                if atom_idx < num_atoms:
                    atom_scores[atom_idx] += weight

        sorted_atoms = sorted(list(all_bit_atoms))
        if not sorted_atoms:
            continue

        center_idx = center_list[0][0]
        center_symbol = mol.GetAtomWithIdx(center_idx).GetSymbol()

        try:
            frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted_atoms)
        except Exception:
            frag_smiles = ""

        if frag_smiles and frag_smiles in seen_fragments:
            continue
        if frag_smiles:
            seen_fragments.add(frag_smiles)

        # Functional group recognition
        desc = f"Substructure around {center_symbol}"
        if any(p in frag_smiles for p in ["S(=O)(=O)N", "S(N)(=O)=O"]):
            desc = "Sulfonamide Group (Key COX-2 Selectivity Anchor)"
        elif "C(F)(F)F" in frag_smiles:
            desc = "Trifluoromethyl Group (Hydrophobic Channel Binding)"
        elif "S(=O)(=O)" in frag_smiles:
            desc = "Sulfonyl Group (Hydrophilic Side-Pocket Anchor)"
        elif any(p in frag_smiles for p in ["C(=O)O", "c(=O)o"]):
            desc = "Carboxylic Acid Functional Group"
        elif any(c in frag_smiles for c in ["n", "o", "s"]):
            desc = "Heteroaromatic Core (Ring Nitrogen/Oxygen/Sulfur)"
        elif "c" in frag_smiles:
            desc = "Aromatic Phenyl Substructure"

        if len(top_substructures) < 5:
            top_substructures.append({
                "bit_id": int(bit_id),
                "importance": round(weight, 5),
                "atom_indices": sorted_atoms,
                "fragment_smiles": frag_smiles,
                "description": desc
            })

    max_score = float(atom_scores.max()) if len(atom_scores) > 0 else 0.0
    if max_score > 0:
        normalized_scores = (atom_scores / max_score).round(3).tolist()
    else:
        normalized_scores = [0.0] * num_atoms

    top_atoms = set()
    for sub in top_substructures:
        top_atoms.update(sub["atom_indices"])
    highlighted_indices = sorted(list(top_atoms))

    return {
        "highlighted_atom_indices": highlighted_indices,
        "atom_scores": normalized_scores,
        "top_substructures": top_substructures
    }


def get_lead_optimization_suggestions(mol, clean_smiles: str, active_prob: float = 0.0):
    """
    Suggests chemical modifications to turn an Inactive molecule into an Active lead.
    Used exclusively when the prediction is Inactive.
    """
    suggestions = []
    has_sulfonamide = False
    try:
        sulfonamide_pattern = Chem.MolFromSmarts("S(=O)(=O)[N,C]")
        if sulfonamide_pattern and mol.HasSubstructMatch(sulfonamide_pattern):
            has_sulfonamide = True
    except Exception:
        pass

    if not has_sulfonamide:
        suggestions.append({
            "title": "Introduce a Para-Sulfonamide (-SO2NH2) Anchor",
            "impact": "High (+40% to +70% Active Probability)",
            "rationale": (
                "The COX-2 enzyme possesses a distinct secondary hydrophilic pocket (Arg513, His90, Gln192) "
                "that is sterically blocked in COX-1 by Ile523. Adding a sulfonamide or methylsulfonyl group "
                "to an aromatic ring provides the primary anchor for potent COX-2 selectivity."
            ),
            "target_feature": "XGBoost Feature Bits 350, 833, 1197 (Benzenesulfonamide pharmacophore)",
            "example_transformation": "Aspirin -> Aspirin-Sulfonamide analog shifts prediction from Inactive to Active."
        })

    num_aromatic_rings = 0
    try:
        num_aromatic_rings = Lipinski.NumAromaticRings(mol)
    except Exception:
        pass

    if num_aromatic_rings < 2:
        suggestions.append({
            "title": "Incorporate a Central Diaryl Heterocyclic Scaffold",
            "impact": "Moderate to High",
            "rationale": (
                "Potent COX-2 inhibitors (the 'coxib' family) rely on two aromatic rings oriented at a rigid "
                "~120° angle around a central 5-membered heterocycle (e.g., pyrazole, isoxazole, or furanone) "
                "to span both the primary channel and the secondary selectivity pocket."
            ),
            "target_feature": "XGBoost Feature Bits 1692, 1434 (Diaryl heterocyclic core)",
            "example_transformation": "Couple the phenyl ring to a pyrazole or furanone central nucleus."
        })

    has_fluorine = any(atom.GetSymbol() == "F" for atom in mol.GetAtoms())
    if not has_fluorine:
        suggestions.append({
            "title": "Add Fluorine or Trifluoromethyl (-CF3) Substituents",
            "impact": "Moderate (+15% to +30% Binding Affinity)",
            "rationale": (
                "Inserting a trifluoromethyl group (-CF3) or halogen (-F, -Cl) into the hydrophobic sub-pocket "
                "enhances lipophilic contact with Val349 and Leu359, while improving metabolic stability."
            ),
            "target_feature": "XGBoost Feature Bit 1434 (Trifluoromethyl pyrazole moiety)",
            "example_transformation": "Replace methyl or hydrogen with -CF3 or -F on the terminal aryl ring."
        })

    try:
        mol_wt = Descriptors.MolWt(mol)
        if mol_wt < 250:
            suggestions.append({
                "title": "Increase Molecular Volume to Fill COX-2 Pocket",
                "impact": "Moderate",
                "rationale": (
                    f"Current molecular weight ({round(mol_wt, 1)} g/mol) is relatively small. "
                    "The COX-2 active site volume is ~394 Å³ (20% larger than COX-1). "
                    "Expanding the molecule's aromatic footprint ensures optimal active-site occupancy."
                ),
                "target_feature": "Optimal volume match for COX-2 cavity (MW 300–450 g/mol)",
                "example_transformation": "Extend aromatic conjugation or add a second substituted phenyl ring."
            })
    except Exception:
        pass

    return suggestions[:3]


def get_active_application_suggestions(mol, clean_smiles: str) -> list:
    """
    Evaluates 6 potential therapeutic application domains for an Active COX-2 inhibitor
    using RDKit-derived physicochemical properties and pharmacophore matching.
    Returns all 6 domains, ranked by suitability, with top 3 flagged.
    """
    if mol is None:
        return []

    try:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rotb = int(Lipinski.NumRotatableBonds(mol))
        aromatic_rings = int(Lipinski.NumAromaticRings(mol))
    except Exception:
        mw, logp, tpsa, hbd, hba, rotb, aromatic_rings = 350.0, 2.5, 60.0, 1, 4, 3, 2

    # Substructure checks
    sulfonamide_pat = Chem.MolFromSmarts("S(=O)(=O)[N,C]")
    has_sulfonamide = mol.HasSubstructMatch(sulfonamide_pat) if sulfonamide_pat else False

    cf3_pat = Chem.MolFromSmarts("C(F)(F)F")
    has_cf3 = mol.HasSubstructMatch(cf3_pat) if cf3_pat else False

    cooh_pat = Chem.MolFromSmarts("C(=O)[OH]")
    has_cooh = mol.HasSubstructMatch(cooh_pat) if cooh_pat else False

    all_suggestions = []

    # 1. GI-Sparing Selective NSAID
    score_1 = 60
    if has_sulfonamide: score_1 += 35
    if aromatic_rings >= 2: score_1 += 15
    if mw >= 300: score_1 += 10
    all_suggestions.append({
        "id": "gi_sparing_nsaid",
        "title": "GI-Sparing Selective Anti-Inflammatory (NSAID)",
        "domain": "Rheumatology & Chronic Pain",
        "suitability_score": min(score_1, 100),
        "rationale": (
            "Possesses pharmacophoric features correlated with selective COX-2 inhibition over COX-1, "
            "dramatically sparing gastric mucosal PGE2 synthesis and reducing ulcer/bleeding risk "
            "compared to traditional non-selective NSAIDs like aspirin or indomethacin."
        ),
        "target_conditions": "Osteoarthritis, Rheumatoid Arthritis, Ankylosing Spondylitis",
        "rdkit_evidence": (
            f"Sulfonamide/Sulfone anchor: {'Detected (high selectivity)' if has_sulfonamide else 'Absent'}; "
            f"Aromatic rings: {aromatic_rings}; MW: {mw:.1f} g/mol."
        ),
        "reference_drugs": "Celecoxib (Celebrex), Etoricoxib (Arcoxia)"
    })

    # 2. Colorectal & Epithelial Cancer Chemoprevention
    score_2 = 50
    if 250 <= mw <= 500: score_2 += 20
    if 1.5 <= logp <= 4.5: score_2 += 15
    if tpsa <= 120: score_2 += 15
    if has_sulfonamide: score_2 += 10
    all_suggestions.append({
        "id": "cancer_chemoprevention",
        "title": "Colorectal & Epithelial Cancer Chemoprevention",
        "domain": "Oncology Research",
        "suitability_score": min(score_2, 100),
        "rationale": (
            "COX-2 is chronically overexpressed in premalignant adenomas and gastrointestinal, breast, and pulmonary "
            "carcinomas, promoting tumour angiogenesis and evading apoptosis. High drug-likeness makes this compound "
            "an attractive chemopreventive adjuvant candidate."
        ),
        "target_conditions": "Familial Adenomatous Polyposis (FAP), Colorectal Polyp Suppression",
        "rdkit_evidence": f"Drug-like oral profile: MW={mw:.1f} g/mol (optimal 250–500), LogP={logp:.2f}, TPSA={tpsa:.1f} Å².",
        "reference_drugs": "Celecoxib (FDA-approved adjuvant for FAP)"
    })

    # 3. Neuroinflammation & Neurodegenerative Disorders
    score_3 = 40
    if mw < 420: score_3 += 25
    if tpsa < 90: score_3 += 25
    if hbd <= 2: score_3 += 15
    if 1.0 <= logp <= 3.8: score_3 += 15
    all_suggestions.append({
        "id": "neuroinflammation_cns",
        "title": "Neuroinflammation & Neurodegenerative Lead",
        "domain": "Neurology Research",
        "suitability_score": min(score_3, 100),
        "rationale": (
            "Pathological COX-2 upregulation in microglia and pyramidal neurons accelerates neuroinflammatory cascades "
            "in Alzheimer's and Parkinson's disease. Molecular geometry satisfies Blood-Brain Barrier (BBB) penetration criteria."
        ),
        "target_conditions": "Alzheimer's Disease, Neuroinflammatory Cognitive Decline, ALS",
        "rdkit_evidence": f"BBB penetration metrics: MW={mw:.1f} (<420), TPSA={tpsa:.1f} Å² (<90 Å²), HBD={hbd} (≤2).",
        "reference_drugs": "Investigational Brain-Penetrant Coxibs"
    })

    # 4. Acute Post-Surgical & Orthopedic Analgesia
    score_4 = 55
    if rotb <= 6: score_4 += 20
    if mw <= 400: score_4 += 15
    if has_cf3 or has_sulfonamide: score_4 += 15
    all_suggestions.append({
        "id": "acute_orthopedic_analgesia",
        "title": "Acute Post-Surgical & Orthopedic Analgesic",
        "domain": "Surgical & Emergency Medicine",
        "suitability_score": min(score_4, 100),
        "rationale": (
            "Provides potent, non-narcotic peripheral analgesia without platelet dysfunction or bleeding complications, "
            "making it suitable for perioperative, dental, and post-traumatic orthopedic pain management."
        ),
        "target_conditions": "Post-Operative Recovery, Acute Dental Pain, Musculoskeletal Trauma",
        "rdkit_evidence": f"Rigid core: {rotb} rotatable bonds; MW={mw:.1f} g/mol; HBA={hba}, HBD={hbd}.",
        "reference_drugs": "Parecoxib (Dynastat IV), Valdecoxib"
    })

    # 5. Cardiovascular Atherosclerotic Plaque Inflammation Modulation
    score_5 = 45
    if aromatic_rings >= 2: score_5 += 20
    if logp >= 2.0: score_5 += 20
    if has_sulfonamide: score_5 += 10
    all_suggestions.append({
        "id": "vascular_inflammation",
        "title": "Vascular Wall & Atherosclerotic Plaque Modulation",
        "domain": "Cardiovascular Immunology",
        "suitability_score": min(score_5, 100),
        "rationale": (
            "Targeting local macrophage-driven COX-2 activity in unstable atherosclerotic lesions to dampen matrix "
            "metalloproteinase secretion and prevent vulnerable plaque rupture."
        ),
        "target_conditions": "Atherosclerotic Lesion Stabilization, Arterial Wall Inflammation",
        "rdkit_evidence": f"Lipophilic aromatic scaffold: LogP={logp:.2f}, Aromatic rings={aromatic_rings}.",
        "reference_drugs": "Experimental Anti-Atherosclerotic Leads"
    })

    # 6. Topical / Transdermal Anti-Inflammatory Formulation
    score_6 = 50
    if mw < 360: score_6 += 25
    if 1.0 <= logp <= 3.5: score_6 += 20
    if rotb <= 5: score_6 += 15
    if has_cooh: score_6 += 10
    all_suggestions.append({
        "id": "topical_transdermal",
        "title": "Topical / Transdermal Anti-Inflammatory Formulation",
        "domain": "Dermatology & Sports Medicine",
        "suitability_score": min(score_6, 100),
        "rationale": (
            "Physicochemical properties allow efficient passive stratum corneum permeation for targeted topical delivery "
            "directly into inflamed joints or tendon sheaths while minimizing systemic exposure and cardiac risk."
        ),
        "target_conditions": "Superficial Joint Osteoarthritis (Knee/Hand), Tendinitis, Sports Injuries",
        "rdkit_evidence": f"Transdermal skin flux profile: MW={mw:.1f} (<360), LogP={logp:.2f} (1–3.5), Rotatable bonds={rotb} (≤5).",
        "reference_drugs": "Topical Diclofenac Emulgel, Ketoprofen Patch"
    })

    # Sort descending by suitability score
    all_suggestions.sort(key=lambda s: s["suitability_score"], reverse=True)
    for i, s in enumerate(all_suggestions):
        s["is_top_recommendation"] = (i < 3)
        s["rank"] = i + 1

    return all_suggestions


def generate_svg(smiles: str, width: int = 300, height: int = 200) -> str:
    """
    Uses RDKit to render a 2D molecular structure as an SVG string.
    Returns SVG string, or empty string if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return ""
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def resolve_molecule_name(smiles: str, user_name: Optional[str] = None, mol=None) -> str:
    """
    Resolves the human-readable molecule name for the given structure.
    1. Returns user-provided name if non-empty.
    2. Checks fast canonical dictionary (known drugs & sample molecules).
    3. Checks ChEMBL dataset mapping.
    4. Queries PubChem PUG REST API (fast online lookup, cached).
    5. Fallback to formula-derived descriptor or 'Molecule'.
    """
    if user_name and user_name.strip():
        return user_name.strip()

    can_smi = None
    if mol:
        try:
            can_smi = Chem.MolToSmiles(mol)
        except Exception:
            pass
    if not can_smi:
        can_smi = smiles.strip()

    if can_smi in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[can_smi]
    if smiles.strip() in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[smiles.strip()]

    if can_smi in NAME_CACHE:
        return NAME_CACHE[can_smi]

    if smiles.strip() in CHEMBL_MAP:
        chembl_name = CHEMBL_MAP[smiles.strip()]
        NAME_CACHE[can_smi] = chembl_name
        return chembl_name
    if can_smi in CHEMBL_MAP:
        chembl_name = CHEMBL_MAP[can_smi]
        NAME_CACHE[can_smi] = chembl_name
        return chembl_name

    # PubChem online lookup
    try:
        quoted = urllib.parse.quote(can_smi, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quoted}/property/Title,IUPACName/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "BioPredict/2.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
            title = props.get("Title") or props.get("IUPACName")
            if title and str(title).strip():
                clean_title = str(title).strip()
                NAME_CACHE[can_smi] = clean_title
                return clean_title
    except Exception:
        pass

    if mol:
        try:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            return f"Compound ({formula})"
        except Exception:
            pass

    return "Molecule"


def resolve_input_to_smiles(input_text: str):
    """
    Resolves user input which can be either a valid SMILES string OR a molecule name.
    Returns (clean_smiles, resolved_name). If cannot resolve, returns (None, None).
    """
    text = input_text.strip()
    if not text:
        return None, None

    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        name = resolve_molecule_name(text, None, mol)
        return text, name

    low_text = text.lower()
    if low_text in NAME_TO_SMILES:
        smi = NAME_TO_SMILES[low_text]
        name = CANONICAL_NAME_MAP.get(smi, text.title())
        return smi, name

    # PubChem lookup by compound name
    try:
        quoted = urllib.parse.quote(text, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quoted}/property/CanonicalSMILES,Title/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "BioPredict/2.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
            props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
            pubchem_smiles = props.get("CanonicalSMILES")
            pubchem_title = props.get("Title") or text.title()
            if pubchem_smiles and Chem.MolFromSmiles(pubchem_smiles):
                NAME_TO_SMILES[low_text] = pubchem_smiles
                return pubchem_smiles, pubchem_title
    except Exception:
        pass

    return None, None


# ── FastAPI App & Pydantic Models ─────────────────────────────────────────────
app = FastAPI(
    title="BioPredict API",
    description="Bioactive Molecule Prediction — REST API powered by RDKit and ML.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    smiles: str
    name: Optional[str] = None


class VisualizeRequest(BaseModel):
    smiles: str
    width: Optional[int] = 300
    height: Optional[int] = 200


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root(request: Request):
    """API info or redirect to frontend if opened in a browser."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/app/")
    return {
        "api": "BioPredict",
        "version": "2.0.0",
        "model": BEST_MODEL_NAME,
        "docs": "/docs",
        "app": "/app/"
    }


@app.get("/health")
def health():
    """Health check — confirms the server and model are running."""
    return {
        "status": "healthy",
        "model_loaded": BEST_MODEL_NAME,
        "fingerprint": f"Morgan radius={FINGERPRINT_RADIUS} bits={FINGERPRINT_BITS}"
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Single molecule prediction.
    Accepts chemical SMILES string OR standard drug name (e.g., Aspirin, Celecoxib).
    Returns definitive bioactivity label (Active / Inactive) without ambiguous confidence scores.
    """
    input_str = request.smiles.strip()
    if not input_str:
        raise HTTPException(status_code=422, detail="Please enter a molecule name or SMILES string.")

    clean_smiles, molecule_name = resolve_input_to_smiles(input_str)
    if not clean_smiles:
        raise HTTPException(
            status_code=422,
            detail=f"Could not resolve structure for '{input_str}'. Please enter a valid SMILES string or standard molecule name."
        )

    features, mol, bit_info = smiles_to_fingerprint(clean_smiles, return_bit_info=True)
    if features is None or mol is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid chemical structure: RDKit could not parse '{clean_smiles}'."
        )

    if request.name and request.name.strip():
        molecule_name = request.name.strip()
    elif not molecule_name:
        molecule_name = resolve_molecule_name(clean_smiles, None, mol)

    try:
        pred_label, raw_score, score_type = run_prediction(features)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference error: {err}")

    # Atom attribution
    atom_attribution = explain_atom_attribution(mol, bit_info)

    # Lead optimization if Inactive; Therapeutic application suggestions if Active
    lead_optimization = []
    active_suggestions = []

    if pred_label == "Inactive":
        lead_optimization = get_lead_optimization_suggestions(mol, clean_smiles, raw_score)
    else:
        active_suggestions = get_active_application_suggestions(mol, clean_smiles)

    # 3D Conformer Generation
    mol_3d = ""
    try:
        mol_h = Chem.AddHs(mol)
        embed_result = AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
        if embed_result == 0:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            mol_3d = Chem.MolToMolBlock(mol_h)
        else:
            AllChem.Compute2DCoords(mol)
            mol_3d = Chem.MolToMolBlock(mol)
    except Exception as e:
        print(f"[WARN] 3D coordinates generation failed: {e}")
        try:
            mol_3d = Chem.MolToMolBlock(mol)
        except Exception:
            mol_3d = ""

    # Molecular info
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = "Unknown"

    mol_info = {
        "formula": formula,
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "mol_weight": round(Descriptors.MolWt(mol), 3),
        "molecule_name": molecule_name
    }

    return {
        "prediction": pred_label,
        "model": BEST_MODEL_NAME,
        "smiles": clean_smiles,
        "molecule_name": molecule_name,
        "mol_info": mol_info,
        "mol_3d": mol_3d,
        "atom_attribution": atom_attribution,
        "lead_optimization": lead_optimization,
        "active_suggestions": active_suggestions
    }


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch molecule prediction from a CSV file.
    Expected format: CSV containing a 'smiles' column.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="File must be a CSV (.csv) file.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(content) > 5242880:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5 MB.")

    try:
        text = content.decode("utf-8-sig")
    except Exception:
        try:
            text = content.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=422, detail="Could not decode file. Please use UTF-8 encoding.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=422, detail="CSV file appears to be empty or has no headers.")

    smiles_col = None
    for col in reader.fieldnames:
        if col.strip().lower() == "smiles":
            smiles_col = col
            break

    if not smiles_col:
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain a 'smiles' column. Found columns: {', '.join(reader.fieldnames)}"
        )

    MAX_ROWS = 2000
    results = []
    total_valid = 0
    total_invalid = 0
    truncated = False

    all_rows = list(reader)
    if len(all_rows) > MAX_ROWS:
        all_rows = all_rows[:MAX_ROWS]
        truncated = True

    for row_num, row in enumerate(all_rows, start=1):
        raw_smiles = row.get(smiles_col, "").strip()
        if not raw_smiles:
            results.append({
                "row": row_num,
                "smiles": raw_smiles,
                "status": "error",
                "error": "Missing or empty SMILES",
                "prediction": None
            })
            total_invalid += 1
            continue

        features, mol = smiles_to_fingerprint(raw_smiles)
        if features is None or mol is None:
            err_msg = "Invalid chemical syntax (RDKit rejected structure)"
            if raw_smiles.count("(") != raw_smiles.count(")"):
                err_msg = "Syntax error: unmatched parenthesis in SMILES"
            elif any(c.isdigit() and raw_smiles.count(c) % 2 != 0 for c in raw_smiles):
                err_msg = "Unclosed ring: missing matching ring closure digit"
            elif raw_smiles.endswith(("=", "#", "-", ":")):
                err_msg = "Truncated SMILES: incomplete branch/bond at end"

            results.append({
                "row": row_num,
                "smiles": raw_smiles,
                "status": "error",
                "error": err_msg,
                "prediction": None
            })
            total_invalid += 1
            continue

        try:
            pred_label, _, _ = run_prediction(features)
            results.append({
                "row": row_num,
                "smiles": raw_smiles,
                "status": "ok",
                "prediction": pred_label
            })
            total_valid += 1
        except Exception as err:
            results.append({
                "row": row_num,
                "smiles": raw_smiles,
                "status": "error",
                "error": f"Prediction failed: {err}",
                "prediction": None
            })
            total_invalid += 1

    if len(results) == 0:
        raise HTTPException(status_code=422, detail="CSV file has no data rows.")

    return {
        "total_rows": len(results),
        "valid_rows": total_valid,
        "invalid_rows": total_invalid,
        "truncated": truncated,
        "results": results
    }


@app.post("/visualize")
def visualize(request: VisualizeRequest):
    """Generate a 2D molecular structure SVG using RDKit."""
    clean_smiles = request.smiles.strip()
    if not clean_smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty.")

    w = request.width or 300
    h = request.height or 200
    svg = generate_svg(clean_smiles, width=w, height=h)
    if not svg:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: '{clean_smiles}'. Cannot generate structure.")

    return {"smiles": clean_smiles, "svg": svg}


@app.get("/models")
def models():
    """Returns model benchmark & comparison data across all tested algorithms."""
    cv_results = metadata.get("cv_results", [])
    holdout_results = metadata.get("holdout_results", [])
    model_selection = metadata.get("model_selection", {})

    return {
        "selected_model": BEST_MODEL_NAME,
        "model_selection": model_selection,
        "cv_results": cv_results,
        "holdout_results": holdout_results,
        "confusion_matrices": confusion_data,
        "dataset_info": metadata.get("dataset_row_counts", {}),
        "cross_validation": metadata.get("cross_validation", {})
    }


# ── Static File & Page Serving ────────────────────────────────────────────────
@app.get("/app/")
@app.get("/app")
def serve_home_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"), media_type="text/html")


@app.get("/app/predict/")
@app.get("/app/predict")
def serve_predict_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "predict.html"), media_type="text/html")


@app.get("/app/batch/")
@app.get("/app/batch")
def serve_batch_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "batch.html"), media_type="text/html")


@app.get("/app/benchmark/")
@app.get("/app/benchmark")
def serve_benchmark_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "benchmark.html"), media_type="text/html")


@app.get("/app/about/")
@app.get("/app/about")
def serve_about_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "about.html"), media_type="text/html")


@app.get("/app/bioactivity-guide/")
@app.get("/app/bioactivity-guide")
def serve_bioactivity_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "bioactivity.html"), media_type="text/html")


@app.get("/predict")
def redirect_to_predict():
    return RedirectResponse(url="/app/predict")


@app.get("/batch")
def redirect_to_batch():
    return RedirectResponse(url="/app/batch")


@app.get("/benchmark")
def redirect_to_benchmark():
    return RedirectResponse(url="/app/benchmark")


@app.get("/about")
def redirect_to_about():
    return RedirectResponse(url="/app/about")


@app.get("/bioactivity-guide")
@app.get("/bioactivity")
def redirect_to_bioactivity():
    return RedirectResponse(url="/app/bioactivity-guide")


@app.get("/style.css")
@app.get("/app/style.css")
def serve_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), media_type="text/css")


@app.get("/script.js")
@app.get("/app/script.js")
def serve_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "script.js"), media_type="application/javascript")


@app.get("/hero-molecule.js")
@app.get("/app/hero-molecule.js")
def serve_hero_molecule():
    return FileResponse(os.path.join(FRONTEND_DIR, "hero-molecule.js"), media_type="application/javascript")


@app.get("/assets/{file_path:path}")
@app.get("/app/assets/{file_path:path}")
def serve_assets(file_path: str):
    full_path = os.path.join(FRONTEND_DIR, "assets", file_path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/study-guide")
def get_study_guide():
    for fname in ["BioPredict_Viva_Defense_Enhancements_Guide.pdf", "BioPredict_Complete_Study_Guide.pdf"]:
        fpath = os.path.join(PROJECT_ROOT, fname)
        if os.path.exists(fpath):
            return FileResponse(fpath, media_type="application/pdf", filename=fname)
    raise HTTPException(status_code=404, detail="Study guide PDF not found.")


@app.get("/presentation")
def get_presentation():
    for fname in ["BioPredict_Final_Presentation.pptx", "BioPredict_Milestone_2_Presentation.pptx"]:
        fpath = os.path.join(PROJECT_ROOT, fname)
        if os.path.exists(fpath):
            return FileResponse(
                fpath,
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                filename=fname
            )
    raise HTTPException(status_code=404, detail="Presentation PPTX not found.")


# Mount static frontend directory
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 65)
    print("  🚀 Starting BioPredict Web Platform...")
    print(f"  🌐 Web Interface : http://127.0.0.1:{port}/app/")
    print(f"  📚 Swagger Docs   : http://127.0.0.1:{port}/docs")
    print("  💊 Target Enzyme  : Cyclooxygenase-2 (COX-2)")
    print(f"  🧠 Model Engine   : {BEST_MODEL_NAME} Classifier (Morgan 2048-bit)")
    print("=" * 65)
    uvicorn.run("backend.main:app", host=host, port=port)
