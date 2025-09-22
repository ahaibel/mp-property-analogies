import itertools as it
import json
import pandas as pd
from llm_inference import run_inference
from pymatgen.core.composition import Composition
from tqdm import tqdm


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


def main_loop(dataset, ref_formula, chem_property, model):
    df = pd.read_csv(f"datasets/{dataset}")
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

    count = 0
    for ref_dict in tqdm(ref_power_set, desc = "Querying with data combinations"):
        trimmed_df = conditional_df(df, ref_dict)[out_cols]
        output = run_inference(trimmed_df, ref_formula, chem_property, model)
        with open (f"output_analogies/{ref_formula}_{chem_property}_{model}_{count}.jsonl", "a", encoding = "utf-8") as f:
            f.write(output.model_dump_json(indent=2) + "\n")

if __name__ == "__main__":
    main_loop()