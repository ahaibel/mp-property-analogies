import pandas as pd

from api_key import MATERIALS_PROJECT_API_KEY as mp_api_key
from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from math import inf
from tqdm import tqdm

SM_LTOL   = 0.35
SM_STOL   = 0.60
SM_ANGTOL = 10.0
RMS_MAX   = 0.30
ANONYMOUS = True


def materials_project_downloads(mp_id: str) -> tuple[int, str, Structure, list]:
    with MPRester(api_key = mp_api_key) as mpr:
        reference_summary = mpr.materials.summary.search(
            material_ids = [mp_id],
            fields = [
                "formula_anonymous",
                "structure",
                "symmetry"
                ]
            )[0]
        reference_formula_anonymous = getattr(reference_summary, "formula_anonymous", None)
        reference_space_group = getattr(reference_summary.symmetry, "number", None)
        reference_structure = getattr(reference_summary, "structure", None)
        candidate_materials = mpr.materials.summary.search(
            spacegroup_number = reference_space_group
            )
    return reference_space_group, reference_formula_anonymous, reference_structure, candidate_materials


def norm_struct(s: Structure, symprec = 1e-2, angle_tol = 10) -> Structure:
    try:
        s_conv = SpacegroupAnalyzer(
            s,
            symprec=symprec,
            angle_tolerance=angle_tol
            ).get_conventional_standard_structure()
        return s_conv.get_sorted_structure()
    except ValueError:
        s_prim = s.get_primitive_structure()
        return s_prim.get_sorted_structure()


def structure_comparisions_to_csv(
    mp_id: str,
    reference_space_group: int,
    reference_formula_anonymous: str,
    reference_structure: Structure,
    candidate_materials: list
    ):
    """
    Filters previously downloaded groups via Pymatgen for structural similarity and exports a CSV.
    There will typically be some number of materials (2~30) that won't be exact matches.
    These will be at the bottom of the CSV, so manually check and cull as needed.
    """
    sm = StructureMatcher(
        ltol=SM_LTOL,
        stol=SM_STOL,
        angle_tol=SM_ANGTOL,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True
    )

    normalized_structure = norm_struct(reference_structure)
    rows = []
    for doc in tqdm(candidates, desc = "Comparing structures"):
        mid   = getattr(doc, "material_id", None)
        form  = getattr(doc, "formula_pretty", None)
        entha = getattr(doc, "formation_energy_per_atom", None)
        bandg = getattr(doc, "band_gap", None)
        s     = getattr(doc, "structure", None)
        if s is None:
            continue
        cand = norm_struct(s)

        if ANONYMOUS:
            is_fit = sm.fit_anonymous(reference_structure, cand)
        else:
            is_fit = sm.fit(reference_structure, cand)
        
        rms = _rms_from_matcher(sm, reference_structure, cand, anonymous=ANONYMOUS)
        lat = cand.lattice
        rows.append({
            "material_id": mid,
            "formula_pretty": form,
            "is_fit": bool(is_fit),
            "rms_A": inf if rms is None else rms,
            "a_A": float(lat.a),
            "b_A": float(lat.b),
            "c_A": float(lat.c),
            "volume_A3": float(lat.volume),
            "band_gap": bandg,
            "formation_energy_per_atom": entha
        })
    
    df = pd.DataFrame(rows).sort_values(["rms_A"])
    analogues = df[(df["rms_A"] <= RMS_MAX) | (df["is_fit"])].copy()
    print(f"{len(analogues)} analogues with (RMS ≤ {RMS_MAX} Å) or StructureMatcher fit=True")
    analogues.to_csv(f"datasets/{reference_space_group}_{reference_formula_anonymous}_{mp_id}.csv", index=False)


def _rms_from_matcher(sm, a, b, anonymous: bool):
    """
    Return RMS Å if available; tolerate tuple/float/None across pmg versions.
    """
    try:
        if anonymous and hasattr(sm, "get_rms_anonymous"):
            res = sm.get_rms_anonymous(a, b)
        else:
            res = sm.get_rms_dist(a, b)
        if res is None:
            return None
        return float(res[0] if isinstance(res, (tuple, list)) else res)
    except Exception:
        return None


if __name__ == "__main__":
    testing_mp_id = "mp-" + input(str("Enter a material id: mp-"))
    group, formula_anonymous, structure, candidates = materials_project_downloads(testing_mp_id)
    structure_comparisions_to_csv(testing_mp_id, group, formula_anonymous, structure, candidates)