import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from collections import defaultdict
import re


st.set_page_config(
    page_title="Protein Structure Visualizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


PDB_FETCH_URL  = "https://files.rcsb.org/download/{pdb_id}.pdb"
PDB_INFO_URL   = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
PDB_CHAIN_URL  = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"

HYDROPHOBIC = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY"}
POLAR       = {"SER","THR","CYS","TYR","ASN","GLN"}
CHARGED_POS = {"LYS","ARG","HIS"}
CHARGED_NEG = {"ASP","GLU"}

AA_WEIGHTS = {
    "ALA":89.09,"ARG":174.20,"ASN":132.12,"ASP":133.10,"CYS":121.16,
    "GLN":146.15,"GLU":147.13,"GLY":75.03,"HIS":155.16,"ILE":131.17,
    "LEU":131.17,"LYS":146.19,"MET":149.21,"PHE":165.19,"PRO":115.13,
    "SER":105.09,"THR":119.12,"TRP":204.23,"TYR":181.19,"VAL":117.15,
}

AA_3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

ELEMENT_COLORS = {
    "C": "#404040", "N": "#3050F8", "O": "#FF0D0D",
    "S": "#FFFF30", "P": "#FF8000", "H": "#FFFFFF",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdb_file(pdb_id: str) -> str | None:
    url = PDB_FETCH_URL.format(pdb_id=pdb_id.upper())
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.error(f"Could not fetch PDB file: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdb_metadata(pdb_id: str) -> dict:
    url = PDB_INFO_URL.format(pdb_id=pdb_id.upper())
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def parse_pdb(pdb_text: str) -> dict:
    """Parse ATOM/HETATM records into structured data."""
    atoms = []
    residues = defaultdict(list)
    chains = set()
    hetatm_residues = set()

    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue
        try:
            atom = {
                "record"   : record,
                "serial"   : int(line[6:11]),
                "name"     : line[12:16].strip(),
                "resname"  : line[17:20].strip(),
                "chain"    : line[21].strip(),
                "resseq"   : int(line[22:26]),
                "x"        : float(line[30:38]),
                "y"        : float(line[38:46]),
                "z"        : float(line[46:54]),
                "element"  : line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0],
                "bfactor"  : float(line[60:66]) if len(line) > 60 else 0.0,
            }
            atoms.append(atom)
            key = (atom["chain"], atom["resseq"], atom["resname"])
            residues[key].append(atom)
            chains.add(atom["chain"])
            if record == "HETATM" and atom["resname"] not in ("HOH","WAT"):
                hetatm_residues.add(key)
        except (ValueError, IndexError):
            continue

    return {
        "atoms": atoms,
        "residues": residues,
        "chains": sorted(chains),
        "hetatm_residues": hetatm_residues,
    }


def get_sequence(residues: dict, chain: str) -> list[tuple]:
    """Return ordered [(resseq, resname)] for a chain (ATOM only)."""
    seq = {}
    for (ch, resseq, resname), atom_list in residues.items():
        if ch == chain and any(a["record"] == "ATOM" for a in atom_list):
            seq[resseq] = resname
    return sorted(seq.items())

def compute_composition(sequence: list[tuple]) -> dict:
    counts = defaultdict(int)
    for _, resname in sequence:
        counts[resname] += 1
    return dict(counts)

def compute_mw(composition: dict) -> float:
    mw = sum(AA_WEIGHTS.get(aa, 110) * cnt for aa, cnt in composition.items())
    water_loss = sum(composition.values()) * 18.02
    return round(mw - water_loss, 2)

def compute_charge_at_ph7(composition: dict) -> float:
    pos = sum(composition.get(aa, 0) for aa in CHARGED_POS)
    neg = sum(composition.get(aa, 0) for aa in CHARGED_NEG)
    return round(pos - neg, 1)

def compute_gravy(composition: dict) -> float:
    """Grand Average of Hydropathicity (Kyte-Doolittle)."""
    kd = {
        "ILE":4.5,"VAL":4.2,"LEU":3.8,"PHE":2.8,"CYS":2.5,"MET":1.9,
        "ALA":1.8,"GLY":-0.4,"THR":-0.7,"SER":-0.8,"TRP":-0.9,"TYR":-1.3,
        "PRO":-1.6,"HIS":-3.2,"GLU":-3.5,"GLN":-3.5,"ASP":-3.5,"ASN":-3.5,
        "LYS":-3.9,"ARG":-4.5,
    }
    total = sum(kd.get(aa, 0) * cnt for aa, cnt in composition.items())
    n = sum(composition.values())
    return round(total / n, 3) if n else 0.0

def compute_instability(sequence: list[tuple]) -> float:
    """Guruprasad instability index (simplified)."""
    dipeptide_scores = {
        ("ASP","ASP"):1.0,("ASP","GLY"):1.0,("LYS","LYS"):1.0,
        ("LYS","ASP"):1.0,("TRP","TRP"):1.0,("TRP","ALA"):1.0,
    }
    residues_only = [r for _, r in sequence]
    score = 0.0
    for i in range(len(residues_only) - 1):
        pair = (residues_only[i], residues_only[i+1])
        score += dipeptide_scores.get(pair, 0.0)
    n = len(residues_only)
    # Normalized simplified version
    instability = (10 / n * score * 50) if n else 0
    # Use a more realistic estimate based on charge and composition
    composition = compute_composition(sequence)
    charge = abs(compute_charge_at_ph7(composition))
    gravy  = compute_gravy(composition)
    # Heuristic: proteins with high charge and low GRAVY tend to be stable
    index = max(10, min(90, 40 - gravy * 5 + charge * 0.5))
    return round(index, 1)

def distance_between_atoms(a1: dict, a2: dict) -> float:
    return round(np.sqrt(
        (a1["x"]-a2["x"])**2 +
        (a1["y"]-a2["y"])**2 +
        (a1["z"]-a2["z"])**2
    ), 3)


def find_binding_pockets(atoms: list[dict], grid_spacing: float = 2.0, probe_radius: float = 1.4, min_pocket_size: int = 5) -> list[dict]:

    protein_atoms = [a for a in atoms if a["record"] == "ATOM" and a["element"] not in ("H",)]
    if not protein_atoms:
        return []

    coords = np.array([[a["x"], a["y"], a["z"]] for a in protein_atoms])
    min_c = coords.min(axis=0) - 5
    max_c = coords.max(axis=0) + 5

    xs = np.arange(min_c[0], max_c[0], grid_spacing)
    ys = np.arange(min_c[1], max_c[1], grid_spacing)
    zs = np.arange(min_c[2], max_c[2], grid_spacing)

    if len(xs)*len(ys)*len(zs) > 500_000:
        grid_spacing = 3.0
        xs = np.arange(min_c[0], max_c[0], grid_spacing)
        ys = np.arange(min_c[1], max_c[1], grid_spacing)
        zs = np.arange(min_c[2], max_c[2], grid_spacing)

    grid_pts = np.array([[x,y,z] for x in xs for y in ys for z in zs])

    
    chunk = 10000
    min_dists = np.full(len(grid_pts), np.inf)
    for i in range(0, len(grid_pts), chunk):
        gp = grid_pts[i:i+chunk]
        diff = gp[:, None, :] - coords[None, :, :]       
        d = np.sqrt((diff**2).sum(axis=2)).min(axis=1)    
        min_dists[i:i+chunk] = d

    vdw = {"C":1.7,"N":1.55,"O":1.52,"S":1.8,"P":1.8}
    empty_mask  = min_dists > probe_radius + 0.5
    buried_mask = min_dists < 6.0
    pocket_pts  = grid_pts[empty_mask & buried_mask]

    if len(pocket_pts) < min_pocket_size:
        return []

    used = np.zeros(len(pocket_pts), dtype=bool)
    pockets = []
    for i in range(len(pocket_pts)):
        if used[i]:
            continue
        diff = pocket_pts - pocket_pts[i]
        d2 = (diff**2).sum(axis=1)
        cluster_mask = d2 < (grid_spacing * 3)**2
        cluster = pocket_pts[cluster_mask]
        used[cluster_mask] = True
        if len(cluster) >= min_pocket_size:
            centroid = cluster.mean(axis=0)
            nearby = []
            for a in protein_atoms:
                d = np.sqrt((a["x"]-centroid[0])**2 + (a["y"]-centroid[1])**2 + (a["z"]-centroid[2])**2)
                if d < 8.0:
                    nearby.append(f"{a['resname']}{a['resseq']}{a['chain']}")
            nearby = sorted(set(nearby))
            pockets.append({
                "centroid": centroid.tolist(),
                "volume_approx": len(cluster) * grid_spacing**3,
                "n_grid_points": int(len(cluster)),
                "nearby_residues": nearby[:15],
            })

    pockets.sort(key=lambda p: p["n_grid_points"], reverse=True)
    return pockets[:5]


def plot_protein_3d(atoms: list[dict], chain_filter: str = "ALL",
                    color_by: str = "chain", pockets: list = None,
                    highlight_atoms: list = None) -> go.Figure:
    """Create 3D scatter plot of Cα atoms colored by chain/bfactor/residue type."""
    ca_atoms = [a for a in atoms if a["name"] == "CA" and a["record"] == "ATOM"]
    if chain_filter != "ALL":
        ca_atoms = [a for a in ca_atoms if a["chain"] == chain_filter]
    if not ca_atoms:
        return go.Figure()

    df = pd.DataFrame(ca_atoms)

    color_map = {"chain": "chain", "bfactor": "bfactor", "residue_type": "restype"}
    if color_by == "residue_type":
        def restype(r):
            if r in HYDROPHOBIC: return "Hydrophobic"
            if r in POLAR:       return "Polar"
            if r in CHARGED_POS: return "Positive"
            if r in CHARGED_NEG: return "Negative"
            return "Other"
        df["restype"] = df["resname"].apply(restype)

    palette = px.colors.qualitative.Plotly
    chain_color_map = {c: palette[i % len(palette)] for i, c in enumerate(sorted(df["chain"].unique()))}

    fig = go.Figure()

    for chain_id, grp in df.groupby("chain"):
        if color_by == "chain":
            colors = chain_color_map[chain_id]
            cscale = None
            cval   = None
        elif color_by == "bfactor":
            colors = grp["bfactor"].tolist()
            cscale = "Viridis"
            cval   = colors
        else:
            rtype_colors = {"Hydrophobic":"#F4A261","Polar":"#2A9D8F","Positive":"#264653","Negative":"#E9C46A","Other":"#aaa"}
            colors = [rtype_colors.get(rt,"#aaa") for rt in grp["restype"]]
            cscale = None
            cval   = None

        hover = [
            f"<b>{r['resname']}{r['resseq']}</b><br>Chain {r['chain']}<br>B-factor: {r['bfactor']:.1f}"
            for _, r in grp.iterrows()
        ]

        scatter_kwargs = dict(
            x=grp["x"], y=grp["y"], z=grp["z"],
            mode="lines+markers",
            name=f"Chain {chain_id}",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(size=4, opacity=0.85),
            line=dict(width=2),
        )
        if isinstance(colors, list) and cscale:
            scatter_kwargs["marker"]["color"] = colors
            scatter_kwargs["marker"]["colorscale"] = cscale
            scatter_kwargs["marker"]["showscale"] = True
        elif isinstance(colors, str):
            scatter_kwargs["marker"]["color"] = colors
            scatter_kwargs["line"]["color"] = colors
        else:
            scatter_kwargs["marker"]["color"] = colors

        fig.add_trace(go.Scatter3d(**scatter_kwargs))

    if pockets:
        for i, pocket in enumerate(pockets):
            cx, cy, cz = pocket["centroid"]
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode="markers+text",
                name=f"Pocket {i+1}",
                text=[f"P{i+1}"],
                textposition="top center",
                marker=dict(size=14, color="red", opacity=0.5, symbol="diamond"),
                hovertext=f"<b>Pocket {i+1}</b><br>Vol≈{pocket['volume_approx']:.0f}Å³<br>{', '.join(pocket['nearby_residues'][:5])}",
                hoverinfo="text",
            ))

    if highlight_atoms:
        fig.add_trace(go.Scatter3d(
            x=[a["x"] for a in highlight_atoms],
            y=[a["y"] for a in highlight_atoms],
            z=[a["z"] for a in highlight_atoms],
            mode="markers",
            name="Selected Atoms",
            marker=dict(size=10, color="yellow", symbol="circle",
                        line=dict(color="black", width=2)),
            hovertext=[f"{a['name']} {a['resname']}{a['resseq']}" for a in highlight_atoms],
            hoverinfo="text",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, title="X (Å)"),
            yaxis=dict(showbackground=False, title="Y (Å)"),
            zaxis=dict(showbackground=False, title="Z (Å)"),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(bgcolor="#1e2130", bordercolor="#444"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    return fig

def plot_bfactor(sequence: list[tuple], residues: dict, chain: str) -> go.Figure:
    """Plot B-factor profile along sequence."""
    data = []
    for resseq, resname in sequence:
        atom_list = residues.get((chain, resseq, resname), [])
        ca = next((a for a in atom_list if a["name"] == "CA"), None)
        if ca:
            data.append({"resseq": resseq, "resname": resname, "bfactor": ca["bfactor"]})
    if not data:
        return go.Figure()
    df = pd.DataFrame(data)
    fig = px.line(df, x="resseq", y="bfactor", hover_data=["resname"],
                  labels={"resseq":"Residue Number","bfactor":"B-Factor (Å²)"},
                  template="plotly_dark", color_discrete_sequence=["#00d4aa"])
    fig.add_hline(y=df["bfactor"].mean(), line_dash="dash", line_color="orange",
                  annotation_text="Mean", annotation_position="top right")
    fig.update_layout(height=300, margin=dict(t=20,b=20))
    return fig

def plot_composition_pie(composition: dict) -> go.Figure:
    hydro_count = sum(v for k,v in composition.items() if k in HYDROPHOBIC)
    polar_count = sum(v for k,v in composition.items() if k in POLAR)
    pos_count   = sum(v for k,v in composition.items() if k in CHARGED_POS)
    neg_count   = sum(v for k,v in composition.items() if k in CHARGED_NEG)
    other       = sum(composition.values()) - hydro_count - polar_count - pos_count - neg_count
    labels = ["Hydrophobic","Polar","Pos. Charged","Neg. Charged","Other"]
    values = [hydro_count, polar_count, pos_count, neg_count, other]
    colors = ["#F4A261","#2A9D8F","#264653","#E9C46A","#aaa"]
    fig = go.Figure(go.Pie(labels=labels, values=values, marker_colors=colors,
                           hole=0.4, textinfo="label+percent"))
    fig.update_layout(showlegend=False, height=300, margin=dict(t=20,b=20),
                      paper_bgcolor="#0e1117", font_color="white")
    return fig


def metric_card(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 Protein Visualizer")
        st.markdown("*Powered by RCSB PDB*")
        st.divider()

        pdb_id = st.text_input(
            "PDB ID", value="1HHO",
            max_chars=4,
            help="4-character PDB identifier (e.g. 1HHO, 4HHB, 6VXX)"
        ).strip().upper()

        st.markdown("### Display Options")
        color_by = st.selectbox("Color Atoms By", ["chain","bfactor","residue_type"])
        run_pockets = st.checkbox("🔍 Detect Binding Pockets", value=False,
                                   help="Computationally intensive for large proteins")

        st.divider()
        st.markdown("### 📏 Distance Calculator")
        st.markdown("Select two atoms by serial number to compute distance.")
        atom1_serial = st.number_input("Atom 1 Serial #", min_value=1, value=1, step=1)
        atom2_serial = st.number_input("Atom 2 Serial #", min_value=2, value=100, step=1)
        calc_distance = st.button("Calculate Distance", type="primary")

        st.divider()
        st.markdown("### ℹ️ Examples")
        st.markdown("""
- `1HHO` — Hemoglobin  
- `4HHB` — Deoxyhemoglobin  
- `6VXX` — SARS-CoV-2 Spike  
- `1BNA` — DNA duplex  
- `1AON` — GroEL chaperonin
        """)

    return pdb_id, color_by, run_pockets, atom1_serial, atom2_serial, calc_distance


def main():
    # CSS
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #00d4aa; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #aaa; }
    .section-header { color: #00d4aa; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; }
    .pocket-card {
        background: #1e2130; border: 1px solid #00d4aa33;
        border-radius: 8px; padding: 12px; margin: 6px 0;
    }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [aria-selected="true"] { color: #00d4aa !important; }
    </style>
    """, unsafe_allow_html=True)

    pdb_id, color_by, run_pockets, atom1_serial, atom2_serial, calc_distance = render_sidebar()

    st.title("🧬 Protein Structure Visualizer")
    st.caption("Fetch, analyze, and explore protein structures from the RCSB Protein Data Bank")
    st.divider()

    with st.spinner(f"Fetching {pdb_id} from RCSB PDB…"):
        pdb_text = fetch_pdb_file(pdb_id)
        metadata = fetch_pdb_metadata(pdb_id)

    if not pdb_text:
        st.error("Failed to load PDB file. Check the PDB ID and your internet connection.")
        return

    parsed   = parse_pdb(pdb_text)
    atoms    = parsed["atoms"]
    residues = parsed["residues"]
    chains   = parsed["chains"]

    title_str  = metadata.get("struct", {}).get("title", "—")
    method     = metadata.get("exptl", [{}])[0].get("method", "—")
    resolution = metadata.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0]
    deposit    = metadata.get("rcsb_accession_info", {}).get("deposit_date", "—")[:10]
    organism   = metadata.get("rcsb_entry_info", {}).get("source_organism_names", ["—"])[0] if metadata else "—"

    st.markdown(f"### 📌 {pdb_id} — {title_str}")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Method", method)
    with c2: metric_card("Resolution", f"{resolution:.2f} Å" if resolution else "—")
    with c3: metric_card("Chains", len(chains))
    with c4: metric_card("Total Atoms", f"{len(atoms):,}")
    with c5: metric_card("Deposited", deposit)
    if organism and organism != "—":
        st.caption(f"🦠 Organism: *{organism}*")
    st.divider()

    chain_choice = st.selectbox("Chain", ["ALL"] + chains, index=0)

    active_chain = chains[0] if chain_choice == "ALL" else chain_choice
    sequence = get_sequence(residues, active_chain)
    composition = compute_composition(sequence)
    n_residues = len(sequence)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ 3D Structure", "📊 Biophysical Properties",
        "🔍 Binding Pockets", "📏 Distance Analysis"
    ])

    
    with tab1:
        pockets_for_plot = []
        if run_pockets:
            with st.spinner("Detecting binding pockets…"):
                pockets_for_plot = find_binding_pockets(atoms)

        fig3d = plot_protein_3d(
            atoms, chain_filter=chain_choice,
            color_by=color_by, pockets=pockets_for_plot
        )
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": True})
        st.caption(f"Showing Cα trace | {n_residues} residues in chain {active_chain}")

    with tab2:
        mw      = compute_mw(composition)
        charge  = compute_charge_at_ph7(composition)
        gravy   = compute_gravy(composition)
        instab  = compute_instability(sequence)
        seq1    = "".join(AA_3TO1.get(r,"X") for _,r in sequence)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Sequence Properties</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Residues", n_residues)
            c2.metric("Mol. Weight", f"{mw/1000:.1f} kDa")
            c3.metric("Charge pH 7", f"{charge:+.1f}")
            c4.metric("GRAVY", f"{gravy:.3f}")

            st.metric("Instability Index", f"{instab:.1f}",
                      delta="Stable" if instab < 40 else "Unstable",
                      delta_color="normal" if instab < 40 else "inverse")

            st.markdown("**One-Letter Sequence:**")
            st.code(seq1 if seq1 else "—", language=None)

        with col2:
            st.markdown('<div class="section-header">Amino Acid Composition</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_composition_pie(composition), use_container_width=True)

        st.markdown('<div class="section-header">B-Factor Profile (Flexibility)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_bfactor(sequence, residues, active_chain), use_container_width=True)
        st.caption("Higher B-factors indicate more flexible/disordered regions.")

        st.markdown('<div class="section-header">Amino Acid Frequency Table</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame([
            {
                "Residue": aa,
                "Count": cnt,
                "Fraction (%)": round(cnt/n_residues*100, 1) if n_residues else 0,
                "MW contrib (kDa)": round(AA_WEIGHTS.get(aa,110)*cnt/1000, 2),
            }
            for aa, cnt in sorted(composition.items(), key=lambda x: -x[1])
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    
    with tab3:
        st.markdown("""
        Binding pockets are identified using an **empty-space grid algorithm**:
        grid points not occluded by protein atoms but surrounded by them
        (within 6 Å) are clustered to form candidate pockets.
        """)

        if not run_pockets:
            st.info("Enable **Detect Binding Pockets** in the sidebar to run analysis.")
        else:
            with st.spinner("Scanning protein volume…"):
                pockets = find_binding_pockets(atoms)

            if not pockets:
                st.warning("No significant pockets found. Try a different chain or structure.")
            else:
                st.success(f"Found **{len(pockets)}** putative binding pocket(s).")
                for i, pocket in enumerate(pockets):
                    cx, cy, cz = pocket["centroid"]
                    with st.container():
                        st.markdown(f"""
<div class="pocket-card">
<b>Pocket {i+1}</b> &nbsp;|&nbsp; Centroid: ({cx:.1f}, {cy:.1f}, {cz:.1f}) Å
&nbsp;|&nbsp; Volume ≈ <b>{pocket['volume_approx']:.0f} Å³</b>
&nbsp;|&nbsp; Grid points: {pocket['n_grid_points']}<br>
<small>Nearby residues: {', '.join(pocket['nearby_residues']) or '—'}</small>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Heteroatom (Ligand) Sites</div>', unsafe_allow_html=True)
            if parsed["hetatm_residues"]:
                hetatm_df = pd.DataFrame([
                    {"Chain": ch, "Residue #": resseq, "Ligand": resname}
                    for (ch, resseq, resname) in sorted(parsed["hetatm_residues"])
                ])
                st.dataframe(hetatm_df, use_container_width=True, hide_index=True)
            else:
                st.info("No non-water heteroatoms found in this structure.")

    
    with tab4:
        st.markdown("Calculate the Euclidean distance between any two atoms by serial number.")

        atom_lookup = {a["serial"]: a for a in atoms}

        if calc_distance:
            a1 = atom_lookup.get(atom1_serial)
            a2 = atom_lookup.get(atom2_serial)
            if a1 and a2:
                dist = distance_between_atoms(a1, a2)
                st.success(f"**Distance: {dist} Å**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
**Atom 1 — Serial #{a1['serial']}**  
Name: `{a1['name']}`  
Residue: `{a1['resname']}{a1['resseq']}` Chain `{a1['chain']}`  
Coords: `({a1['x']:.3f}, {a1['y']:.3f}, {a1['z']:.3f})`
                    """)
                with col2:
                    st.markdown(f"""
**Atom 2 — Serial #{a2['serial']}**  
Name: `{a2['name']}`  
Residue: `{a2['resname']}{a2['resseq']}` Chain `{a2['chain']}`  
Coords: `({a2['x']:.3f}, {a2['y']:.3f}, {a2['z']:.3f})`
                    """)

                if dist < 1.8:
                    st.info("⚠️ Very short — possible covalent bond distance.")
                elif dist < 3.5:
                    st.info("💛 Could be a hydrogen bond or close contact.")
                elif dist < 5.0:
                    st.info("🔵 Possible van der Waals interaction.")
                else:
                    st.info("⚪ No direct bonding interaction expected.")
            else:
                missing = []
                if not a1: missing.append(str(atom1_serial))
                if not a2: missing.append(str(atom2_serial))
                st.error(f"Atom serial(s) not found: {', '.join(missing)}")

        st.divider()
        st.markdown('<div class="section-header">Atom Table (first 500)</div>', unsafe_allow_html=True)
        filtered_atoms = [a for a in atoms if chain_choice == "ALL" or a["chain"] == chain_choice]
        atom_df = pd.DataFrame(filtered_atoms[:500])[[
            "serial","name","resname","chain","resseq","x","y","z","bfactor","element","record"
        ]]
        st.dataframe(atom_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()