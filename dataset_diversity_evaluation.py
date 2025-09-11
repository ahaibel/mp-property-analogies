import pandas as pd
import itertools as it
from pymatgen.core.composition import Composition


def dict_power_set(dictionary):
    items = list(dictionary.items())
    return [
        dict(combo)
        for r in range(len(items) + 1) 
        for combo in it.combinations(items, r)
    ]


def evaluate_element_duplication():
    analogues = pd.read_csv("datasets/" + input(str("datasets/")))
    reference_formula = input(str("Reference formula: "))
    formula_list = analogues["formula_pretty"].tolist()
    reference_composition = Composition(reference_formula).get_el_amt_dict()
    reference_power_set = dict_power_set(reference_composition)

    for element_dict in reference_power_set:
        count = 0
        for formula in formula_list:
            composition = Composition(formula).get_el_amt_dict()
            for element in element_dict.keys():
                if element in composition.keys():
                    if element_dict[element] == composition[element]:
                        count +=1
                        break
        print(f"{element_dict}: {count}")

12
if __name__ == "__main__":
    evaluate_element_duplication()