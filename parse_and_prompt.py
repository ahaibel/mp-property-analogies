import itertools as it
import pandas as pd
import llms
from pymatgen.core.composition import Composition


def dict_power_set(dictionary):
    items = list(dictionary.items())
    return [
        dict(combo)
        for r in range(len(items) + 1) 
        for combo in it.combinations(items, r)
    ]


def conditional_df(df, ref_dict):
    dropped_rows = df["comp"].apply(
        lambda comp: any(comp.get(el) == val for el, val in ref_dict.items())
    )
    return df[~dropped_rows].reset_index(drop=True)


def main_loop():
    ref_formula = "AcClO"
    df = pd.read_csv("datasets/129_ABC_mp-30273.csv")
    chem_property = "volume"
    ref_elements = Composition(ref_formula).get_el_amt_dict()
    ref_power_set = dict_power_set(ref_elements)
    df["comp"] = df["formula_pretty"].apply(lambda f: Composition(f).get_el_amt_dict())
    mask = df["formula_pretty"].apply(lambda f: Composition(f).get_el_amt_dict() != ref_elements)
    df = df[mask].reset_index(drop=True)

    if chem_property == "band_gap":
        out_cols=["formula_pretty", "band_gap"]
    elif chem_property == "formation_energy":
        out_cols=["formula_pretty", "formation_energy_per_atom"]
    elif chem_property == "volume":
        out_cols=["formula_pretty", "a_A", "b_A", "c_A", "volume_A3"]

    for ref_dict in ref_power_set:
        trimmed_df = conditional_df(df, ref_dict)[out_cols]
        print(ref_dict, len(trimmed_df))

if __name__ == "__main__":
    main_loop()