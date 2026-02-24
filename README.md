# 🧬 Protein Structure Visualizer

A Streamlit dashboard for exploring protein structures from the **RCSB Protein Data Bank (PDB)** — no local software required.

---

## Features

| Tab | What it does |
|-----|-------------|
| **3D Structure** | Interactive Cα trace colored by chain, B-factor, or residue type |
| **Biophysical Properties** | MW, charge, GRAVY index, instability index, B-factor profile, AA composition pie |
| **Binding Pockets** | Grid-based empty-space pocket detection + ligand/heteroatom sites |
| **Distance Analysis** | Euclidean distance between any two atoms by serial number, with interaction interpretation |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

1. Enter a **4-character PDB ID** in the sidebar (e.g. `1HHO` for hemoglobin).
2. Pick a **chain** and a **coloring scheme**.
3. (Optional) Enable **Detect Binding Pockets** for pocket analysis.
4. Use the **Distance Calculator** — enter two atom serial numbers and click Calculate.

### Example PDB IDs
| ID | Description |
|----|-------------|
| `1HHO` | Oxyhemoglobin |
| `6VXX` | SARS-CoV-2 Spike Protein |
| `1AON` | GroEL-GroES chaperonin complex |
| `1BNA` | DNA B-form duplex |
| `4HHB` | Deoxyhemoglobin |

---

## Technical Details

### Pocket Detection Algorithm
Uses an **empty-space grid method**:
1. A 3D grid is placed over the protein bounding box.
2. Grid points farther than van der Waals radius from all atoms but within 6 Å of the protein surface are marked as "pocket candidates".
3. Candidates are clustered by proximity; clusters > 5 points are reported as pockets.

### Biophysical Properties
- **Molecular Weight** — sum of AA residue weights minus water loss
- **Net Charge at pH 7** — count of Arg/Lys/His minus Asp/Glu
- **GRAVY Index** — Kyte-Doolittle hydropathicity average
- **Instability Index** — heuristic based on charge and GRAVY (Guruprasad method)
- **B-factor Profile** — per-residue Cα temperature factors (flexibility proxy)

---

## Why This Project Matters

Google DeepMind's AlphaFold revolutionized structural biology by predicting protein 3D structures from sequence alone. This dashboard sits at the intersection of **software engineering** and **natural sciences** by:
- Programmatically accessing the world's largest structural biology database
- Implementing bioinformatics algorithms (pocket detection, physicochemical analysis)
- Delivering results through an interactive, production-quality UI

---

## File Structure

```
protein_visualizer/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
