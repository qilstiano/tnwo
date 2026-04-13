"""Shared publication-quality matplotlib style for TNWO experiment figures."""

import matplotlib as mpl


FAMILY_META = {
    "baseline":      {"color": "#2563eb", "label": "Baseline"},
    "annex_heavy":   {"color": "#dc2626", "label": "Annex-Heavy"},
    "peace_heavy":   {"color": "#059669", "label": "Peace-Heavy"},
    "dense_only":    {"color": "#f59e0b", "label": "Dense-Only"},
    "terminal_only": {"color": "#7c3aed", "label": "Terminal-Only"},
}
FAMILY_ORDER = ["baseline", "annex_heavy", "peace_heavy", "dense_only", "terminal_only"]

CATEGORY_META = {
    "aggression": {"color": "#dc2626", "abbr": "Agg", "label": "aggression"},
    "science":    {"color": "#2563eb", "abbr": "Sci", "label": "science"},
    "economy":    {"color": "#f59e0b", "abbr": "Eco", "label": "economy"},
    "diplomacy":  {"color": "#059669", "abbr": "Dip", "label": "diplomacy"},
}
CATEGORY_ORDER = ["aggression", "science", "economy", "diplomacy"]

SUITE_META = {
    "vs_balanced":  {"label": "Balanced ×4",  "short": "vs-bal", "color": "#2563eb"},
    "vs_aggressor": {"label": "Aggressor ×4", "short": "vs-agg", "color": "#dc2626"},
    "vs_diverse":   {"label": "Diverse mix",  "short": "vs-div", "color": "#7c3aed"},
    "vs_diplomat":  {"label": "Diplomat ×4",  "short": "vs-dip", "color": "#059669"},
}
SUITE_ORDER = ["vs_balanced", "vs_aggressor", "vs_diverse", "vs_diplomat"]


def infer_family(run_name: str) -> str:
    if run_name.startswith("base_"):         return "baseline"
    if run_name.startswith("annex_"):        return "annex_heavy"
    if run_name.startswith("peace_"):        return "peace_heavy"
    if run_name.startswith("dense_only"):    return "dense_only"
    if run_name.startswith("terminal_only"): return "terminal_only"
    return "baseline"


def infer_opponent(run_name: str) -> str:
    """Return a short training-opponent key for a run name."""
    if "200t" in run_name:  return "diverse_200t"
    if "seed1" in run_name: return "diverse_seed1"
    if "seed2" in run_name: return "diverse_seed2"
    for opp in ("balanced", "aggressor", "diverse", "diplomat"):
        if run_name.endswith(opp):
            return opp
    return "other"


# Short, paper-friendly row labels
OPP_LABEL = {
    "balanced":       r"Balanced $\times 4$",
    "aggressor":      r"Aggressor $\times 4$",
    "diverse":        "Diverse mix",
    "diplomat":       r"Diplomat $\times 4$",
    "diverse_200t":   r"Diverse mix ($T{=}200$)",
    "diverse_seed1":  "Diverse mix (seed 1)",
    "diverse_seed2":  "Diverse mix (seed 2)",
}


def sort_key(run_name: str):
    fam = infer_family(run_name)
    fi = FAMILY_ORDER.index(fam)
    opp_order = ["balanced", "aggressor", "diverse",
                 "diverse_seed1", "diverse_seed2", "diverse_200t", "diplomat"]
    opp = infer_opponent(run_name)
    oi = opp_order.index(opp) if opp in opp_order else 99
    return (fi, oi)


def set_pub_style():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset":  "stix",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9.5,
        "ytick.labelsize":    9.5,
        "legend.fontsize":    9.5,
        "axes.linewidth":     0.9,
        "axes.edgecolor":     "#333333",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "axes.grid":          False,
        "figure.dpi":         140,
        "savefig.dpi":        300,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })
