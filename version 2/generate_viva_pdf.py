"""
generate_viva_pdf.py
Generates a comprehensive, professional 3-page Viva Defense PDF Guide:
"BioPredict: New Enhancements Beyond the Base Paper & Viva Defense Guide"
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.abspath("scratch")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

PDF_FILENAME = "BioPredict_Viva_Defense_Enhancements_Guide.pdf"

def create_header(ax, title, subtitle, page_num):
    # Header bar
    rect = patches.FancyBboxPatch((0.04, 0.90), 0.92, 0.08,
                                  boxstyle="round,pad=0.015,rounding_size=0.015",
                                  facecolor="#0f766e", edgecolor="#0d5f58", zorder=2)
    ax.add_patch(rect)
    ax.text(0.06, 0.945, title, fontsize=14, fontweight="bold", color="#ffffff", va="center", zorder=3)
    ax.text(0.06, 0.918, subtitle, fontsize=9.5, color="#cceae6", va="center", zorder=3)
    
    # Page indicator
    ax.text(0.93, 0.935, f"Page {page_num} of 3", fontsize=9, fontweight="bold", color="#ffffff", ha="right", va="center", zorder=3)
    
    # Footer
    ax.axhline(0.04, color="#cbd5e1", linewidth=0.8, xmin=0.04, xmax=0.96)
    ax.text(0.04, 0.025, "BioPredict: Bioactive Molecule Prediction Platform", fontsize=8, color="#64748b", va="center")
    ax.text(0.96, 0.025, "Academic Viva & Defense Study Guide", fontsize=8, color="#64748b", ha="right", va="center")

def draw_card(ax, x, y, w, h, bg_color="#ffffff", border_color="#e2e8f0"):
    box = patches.FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                facecolor=bg_color, edgecolor=border_color, linewidth=1.2, zorder=1)
    ax.add_patch(box)
    return box

def render_page_1(pdf):
    fig = plt.figure(figsize=(8.5, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    create_header(ax, "BIOPREDICT: NEW ENHANCEMENTS BEYOND BASE PAPER",
                  "Comprehensive Viva Defense Guide: Technical Additions, Novelties & Methodologies", 1)

    # 1. Project Background & Base Paper Card
    draw_card(ax, 0.04, 0.74, 0.92, 0.14, "#ffffff", "#cbd5e1")
    ax.text(0.06, 0.855, "1. SCIENTIFIC BACKGROUND & THE BASE PAPER", fontsize=10.5, fontweight="bold", color="#0f766e")
    
    bg_text = (
        "- Target Biological System: Cyclooxygenase-2 (COX-2 / PTGS2), ChEMBL ID CHEMBL230 (6,240 curated compounds).\n"
        "  COX-2 is an inducible enzyme producing pro-inflammatory prostaglandins. Selective inhibition relieves pain/swelling\n"
        "  without disturbing constitutive COX-1, completely avoiding severe gastric ulcers and internal gastrointestinal bleeding.\n"
        "- Base Paper: Mustapha & Saeed, 'Bioactivity Prediction of Cyclooxygenase-2 Inhibitors...', Molecules (MDPI), 2016.\n"
        "- Base Paper Scope: Evaluated standard machine learning algorithms (Support Vector Machines, Artificial Neural Networks,\n"
        "  Naive Bayes, Random Forest) using 10-fold cross-validation and AUC = (Sensitivity + Specificity) / 2 (their Eq. 11).\n"
        "- Project Novelty: BioPredict adopts their benchmarked dataset and evaluation formula as the ground-truth baseline,\n"
        "  but introduces 5 revolutionary engineering and scientific enhancements that the base paper did not have."
    )
    ax.text(0.06, 0.755, bg_text, fontsize=8.2, color="#1e293b", va="top", linespacing=1.35)

    # 2. Side-by-Side Comparison Table
    draw_card(ax, 0.04, 0.38, 0.92, 0.34, "#ffffff", "#cbd5e1")
    ax.text(0.06, 0.695, "2. DIRECT COMPARISON: BASE PAPER (2016) VS. BIOPREDICT PLATFORM (OUR WORK)", fontsize=10.5, fontweight="bold", color="#0f766e")

    # Table layout
    headers = ["Technical Dimension", "Base Paper (Mustapha & Saeed, 2016)", "BioPredict Enhancements (Our Work)"]
    col_x = [0.06, 0.28, 0.61]
    
    # Table header bar
    hdr_box = patches.Rectangle((0.05, 0.655), 0.90, 0.026, facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8, zorder=2)
    ax.add_patch(hdr_box)
    ax.text(col_x[0], 0.668, headers[0], fontsize=8.5, fontweight="bold", color="#0f172a", va="center", zorder=3)
    ax.text(col_x[1], 0.668, headers[1], fontsize=8.5, fontweight="bold", color="#0f172a", va="center", zorder=3)
    ax.text(col_x[2], 0.668, headers[2], fontsize=8.5, fontweight="bold", color="#0f766e", va="center", zorder=3)

    rows = [
        ("Core Classifier", "Linear SVM, basic ANN (MLP), Naive Bayes, RF", "Extreme Gradient Boosting (XGBoost 200 trees, max_depth=16)"),
        ("Molecular Featurization", "Traditional 1D/2D physicochemical descriptors", "2,048-bit Morgan Circular Fingerprints (ECFP4, radius=2)"),
        ("Validation Accuracy", "Reported ~78% to 80% on cross-validation folds", "Achieved 82.21% CV accuracy & 0.7594 balanced ROC-AUC"),
        ("3D Molecular Dynamics", "None (offline calculations, no structures)", "Real-time ETKDG 3D embedding + MMFF94 force-field WebGL viewer"),
        ("Model Interpretability", "Complete 'Black Box' (no explanation of why)", "Explainable AI (XAI) mapping tree splits to atoms with 3D highlighting"),
        ("Actionable Feedback", "Passive binary classification (Active / Inactive)", "Lead Optimization Engine suggesting specific SAR changes for inactives"),
        ("High-Throughput Batch", "None (single offline script in notebook)", "Full Batch Virtual Screening for CSV compound libraries (up to 2,000)"),
        ("Deployment & UI", "Static academic tables, no user interface", "Interactive Multi-Page Web Platform (FastAPI + Modern HTML5/JS)")
    ]

    curr_y = 0.635
    for i, (dim, base, bio) in enumerate(rows):
        bg_row = "#ffffff" if i % 2 == 0 else "#f8fafc"
        row_box = patches.Rectangle((0.05, curr_y - 0.012), 0.90, 0.028, facecolor=bg_row, edgecolor="none", zorder=2)
        ax.add_patch(row_box)
        ax.text(col_x[0], curr_y + 0.002, dim, fontsize=7.8, fontweight="bold", color="#1e293b", va="center", zorder=3)
        ax.text(col_x[1], curr_y + 0.002, base, fontsize=7.6, color="#475569", va="center", zorder=3)
        ax.text(col_x[2], curr_y + 0.002, bio, fontsize=7.6, fontweight="bold", color="#0f766e", va="center", zorder=3)
        ax.axhline(curr_y - 0.012, color="#e2e8f0", linewidth=0.5, xmin=0.05, xmax=0.95, zorder=2)
        curr_y -= 0.029

    # 3. Summary of Key Novelty Callout Box
    draw_card(ax, 0.04, 0.065, 0.92, 0.29, "#f0fdfa", "#99f6e4")
    ax.text(0.06, 0.33, "3. KEY NOVELTY SUMMARY: WHAT MAKES BIOPREDICT UNIQUE IN A VIVA DEFENSE?", fontsize=10.5, fontweight="bold", color="#0f766e")
    
    novelty_points = (
        "1. From Black-Box to Transparent AI: While the base paper just gave a 'yes/no' probability, BioPredict visually shows the chemist\n"
        "   which functional group (e.g. sulfonamide anchor, trifluoromethyl group) triggered the prediction in interactive 3D.\n\n"
        "2. Prescriptive Rather Than Just Descriptive: Typical ML models stop after saying a compound is Inactive. BioPredict includes\n"
        "   a rule-based Lead Optimization Engine that prescribes exact bioisosteric transformations to convert inactive molecules into leads.\n\n"
        "3. True In Silico Biomechanics: Converting 1D SMILES text into an energy-minimized 3D conformer with MMFF94 force-field physics\n"
        "   bridges the gap between abstract machine learning and structural biology.\n\n"
        "4. Zero External AI Dependency: The system does NOT rely on proprietary cloud LLMs (OpenAI, Gemini). All machine learning\n"
        "   (XGBoost), cheminformatics (RDKit), and 3D rendering (3Dmol.js) run 100% locally, deterministically, and reproducibly.\n\n"
        "5. Clean Multi-Page Web Deployment: Converted from a monolithic single page into a robust multi-page application with\n"
        "   FastAPI server-side routing, responsive navigation, batch CSV processing, and dedicated benchmark leaderboards."
    )
    ax.text(0.06, 0.305, novelty_points, fontsize=8.0, color="#134e4a", va="top", linespacing=1.32)

    pdf.savefig(fig)
    plt.close()

def render_page_2(pdf):
    fig = plt.figure(figsize=(8.5, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    create_header(ax, "DEEP DIVE: THE 4 MAJOR TECHNICAL ENHANCEMENTS",
                  "How Each Feature Works Under the Hood & How to Explain Them in Simple Words", 2)

    enhancements = [
        ("ENHANCEMENT 1: 3D MOLECULAR SIMULATION & CONFORMATION ENGINE",
         "#0f766e", "#ffffff", "#cbd5e1",
         "What It Is: Converts a flat 1D SMILES chemical formula into a realistic, rotating 3D molecular structure.\n"
         "How It Works Technically:\n"
         "  1. Explicit Hydrogen Addition (Chem.AddHs): SMILES omits hydrogen; RDKit adds them so tetrahedral 3D bond angles are correct.\n"
         "  2. Distance Geometry (ETKDG): Embeds atoms in (x, y, z) Cartesian space with a fixed random seed (42) for reproducibility.\n"
         "  3. Energy Minimization (MMFF94): Runs Merck Molecular Force Field optimization (50 iterations) to eliminate steric clashes.\n"
         "  4. WebGL Canvas Rendering: Serialized into an SDF MolBlock and rendered in browser memory via 3Dmol.js (Ball & Stick / Stick).\n"
         "Viva Simple Words: 'Instead of just showing a flat formula, our system runs a virtual physics simulation to fold the molecule into\n"
         "its natural lowest-energy 3D shape, allowing chemists to inspect binding angles before ordering expensive lab synthesis.'",
         0.69, 0.185),

        ("ENHANCEMENT 2: EXPLAINABLE AI (XAI) & ATOM ATTRIBUTION",
         "#d97706", "#ffffff", "#cbd5e1",
         "What It Is: Peeks inside the XGBoost 'black box' to highlight the exact atoms and chemical fragments responsible for the verdict.\n"
         "How It Works Technically:\n"
         "  1. bitInfo Capture: During Morgan Fingerprint calculation (radius=2, 2048-bit), RDKit records which center atom activated each bit.\n"
         "  2. XGBoost Learned Weights: The model trained Gini impurity split gains (feature_importances_) across all 2048 fingerprint bits.\n"
         "  3. Atom Mapping: Active bits with positive importance are traced back to their participating atoms via Chem.FindAtomEnvironmentOfRadiusN.\n"
         "  4. Dynamic 3D Highlighting: In the UI, clicking a substructure highlights those exact atoms in vibrant amber (#f59e0b) on the 3D model.\n"
         "Viva Simple Words: 'Every molecule is converted into a 2,048-digit barcode. Our AI learned which digits represent active COX-2\n"
         "inhibitors. Explainable AI traces those winning digits back to the exact atoms and highlights them like a digital highlighter pen.'",
         0.485, 0.185),

        ("ENHANCEMENT 3: LEAD OPTIMIZATION ENGINE FOR INACTIVE COMPOUNDS",
         "#be123c", "#ffffff", "#cbd5e1",
         "What It Is: When a compound is predicted Inactive (e.g. Aspirin), the engine prescribes chemical edits to convert it into an active drug.\n"
         "How It Works Technically:\n"
         "  1. Selectivity Pocket Check: Detects absence of a Sulfonamide (-SO2NH2) or Sulfonyl group required for COX-2's unique side pocket (Arg513).\n"
         "  2. Central Scaffold Rigidification: Checks for a diaryl 5-membered heterocyclic core (pyrazole/furanone) oriented at ~120 degrees.\n"
         "  3. Halogen Substitution: Recommends introducing -F or -CF3 to engage hydrophobic channel contacts (Val349, Leu359).\n"
         "  4. Volume Match: Assesses molecular weight against the ~394 cubic-angstrom COX-2 cavity (20% larger than COX-1).\n"
         "Viva Simple Words: 'The AI knows the exact shape of the COX-2 lock. If your key doesn't fit, the system compares your key to the\n"
         "lock, finds the missing teeth, and tells you: Add a sulfonamide group here to boost activity probability from 10% to 81%.'",
         0.28, 0.185),

        ("ENHANCEMENT 4: HIGH-THROUGHPUT BATCH VIRTUAL SCREENING",
         "#2563eb", "#ffffff", "#cbd5e1",
         "What It Is: Enables pharmaceutical screening of large chemical compound libraries from a CSV file in a single click.\n"
         "How It Works Technically:\n"
         "  1. Automated Column Detection: Intelligently detects 'smiles' or 'SMILES' column headers while preserving custom ID metadata.\n"
         "  2. Fault-Tolerant Processing: Validates up to 2,000 rows. Syntax errors are caught and flagged without crashing the batch run.\n"
         "  3. Real-Time Analytics: Dynamically computes Hit Rate percentage, active/inactive counts, and generates interactive donut charts.\n"
         "  4. Filtered CSV Export: Allows researchers to filter by 'Active' hits and export clean CSV datasets for immediate bench testing.\n"
         "Viva Simple Words: 'A medicinal chemist can upload a spreadsheet of 2,000 newly synthesized compounds, and within seconds the\n"
         "platform filters out all non-binders, computes the hit rate, and exports the top active candidates ready for lab validation.'",
         0.075, 0.185)
    ]

    for title, col, bg, border, text, y_pos, height in enhancements:
        draw_card(ax, 0.04, y_pos, 0.92, height, bg, border)
        ax.text(0.06, y_pos + height - 0.022, title, fontsize=9.5, fontweight="bold", color=col)
        ax.text(0.06, y_pos + height - 0.042, text, fontsize=7.8, color="#1e293b", va="top", linespacing=1.28)

    pdf.savefig(fig)
    plt.close()

def render_page_3(pdf):
    fig = plt.figure(figsize=(8.5, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    create_header(ax, "VIVA VOCE DEFENSE: TOP QUESTIONS & ANSWERS",
                  "Fast-Fire Q&A Cheat Sheet for University / Project Panel Defense", 3)

    qa_list = [
        ("Q1: What was the base paper and why did you choose it?",
         "Answer: Our base paper is Mustapha & Saeed (Molecules, 2016). They published the benchmark methodology for predicting COX-2\n"
         "inhibitors using ChEMBL bioassays (IC50 <= 1000 nM) and defined the balanced metric AUC = (Sensitivity + Specificity) / 2.\n"
         "We chose it as our gold standard because COX-2 is a clinically vital target for pain and inflammation where selectivity is paramount."),

        ("Q2: Why did you pick XGBoost over Support Vector Machines or Deep Learning?",
         "Answer: In our 10-fold cross-validation benchmark across 6 classifiers, XGBoost achieved the highest validation AUC (0.7594) and highest\n"
         "accuracy (82.21%). Gradient-boosted decision trees naturally handle sparse binary Morgan fingerprints (2048 bits) with built-in L1/L2\n"
         "regularization, preventing the severe overfitting and computational overhead that Deep Learning (ANN) suffered from."),

        ("Q3: Isn't Machine Learning a black box? How do you prove your model understands chemistry?",
         "Answer: That is exactly why we implemented Explainable AI (XAI). Using RDKit's bitInfo, we map XGBoost's learned tree split weights\n"
         "back to physical atoms. When tested on Celecoxib, our model autonomously assigns the highest weights to the sulfonamide group\n"
         "(-SO2NH2) and trifluoromethyl pyrazole ring. These are the exact pharmacophores proven in X-ray crystallography to bind COX-2!"),

        ("Q4: How does your Lead Optimization Engine know what will improve an inactive compound?",
         "Answer: It compares the inactive molecule against the learned 3D structural blueprint of the COX-2 enzyme. COX-2 has a distinct secondary\n"
         "pocket that COX-1 lacks. The engine scans for: (1) a sulfonamide anchor, (2) a rigid diaryl core, and (3) halogen channel grips.\n"
         "If missing, it suggests specific bioisosteric transformations. For example, adding a sulfonamide to Aspirin flips its score from 10% to 81%."),

        ("Q5: Did you use ChatGPT, Gemini, or any external AI APIs to make these predictions?",
         "Answer: No external AI was used. BioPredict is 100% self-contained and local. The predictions come from our own pre-trained XGBoost model\n"
         "(best_model.pkl). The fingerprints are computed locally by RDKit. The only network call is an optional PubChem dictionary lookup to\n"
         "convert trade names (like 'Aspirin') into SMILES. The platform can run completely offline in a secure pharmaceutical intranet."),

        ("Q6: How did you generate the 3D molecular structures from a text string?",
         "Answer: We take the 1D SMILES text, add explicit hydrogens using RDKit (Chem.AddHs), and apply the ETKDG distance geometry algorithm\n"
         "(AllChem.EmbedMolecule) to calculate (x, y, z) coordinates. Then we run Merck Molecular Force Field (MMFF94) energy minimization\n"
         "to relieve steric clashes and fold it into its natural conformation, rendered in the browser using WebGL via 3Dmol.js."),

        ("Q7: How is the application architected and deployed?",
         "Answer: BioPredict is structured as a professional multi-page web application. The backend is built with FastAPI and Uvicorn,\n"
         "serving dedicated routes (/app/, /app/predict, /app/batch, /app/benchmark, /app/about). The frontend is built with pure HTML5, CSS3,\n"
         "and vanilla JavaScript, providing sub-second inference, interactive 3D WebGL viewers, and responsive mobile navigation.")
    ]

    curr_y = 0.865
    box_height = 0.108
    gap = 0.009

    for q, a in qa_list:
        draw_card(ax, 0.04, curr_y - box_height, 0.92, box_height, "#ffffff", "#cbd5e1")
        ax.text(0.06, curr_y - 0.022, q, fontsize=8.8, fontweight="bold", color="#0f766e")
        ax.text(0.06, curr_y - 0.042, a, fontsize=7.6, color="#1e293b", va="top", linespacing=1.28)
        curr_y -= (box_height + gap)

    pdf.savefig(fig)
    plt.close()

def main():
    print(f"Generating publication-grade PDF: {PDF_FILENAME}...")
    with PdfPages(PDF_FILENAME) as pdf:
        print(" - Rendering Page 1: Base Paper vs. BioPredict Enhancements...")
        render_page_1(pdf)
        print(" - Rendering Page 2: Deep Dive into 4 Core Technical Enhancements...")
        render_page_2(pdf)
        print(" - Rendering Page 3: Viva Voce Defense Top Q&A Master Sheet...")
        render_page_3(pdf)
    
    file_size = os.path.getsize(PDF_FILENAME)
    print(f"SUCCESS: Generated {PDF_FILENAME} ({file_size / 1024:.1f} KB, 3 pages)")

if __name__ == "__main__":
    main()
