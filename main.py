import argparse
import itertools as it
import pandas as pd
from parse_and_prompt import main_loop
from pymatgen.core.composition import Composition
from grading import Grading

grading = Grading()

def grade_analogy_output(analogy_text, model="gpt-5-mini"):
    grading = Grading()
    scores = grading.llm_grade_analogy(analogy_text, model=model)
    total, percent, letter_grade = grading.compute_score(scores)
    manifest = {
        "analogy": analogy_text,
        "scores": scores,
        "weighted_score": total,
        "percent": percent,
        "letter_grade": letter_grade
    }
    with open("grading_manifest.json", "w") as f:
        import json
        json.dump(manifest, f, indent=2)
    print(" saved to grading_manifest.json.")

property_options = ["band_gap", "formation_energy", "volume"]


def get_arguments():
    parser = argparse.ArgumentParser(description="Arguments for the dataset")
    parser.add_argument(
        "--dataset",
        type = str,
        help = "Dataset to be used from dataset/"
    )
    parser.add_argument(
        "--material",
        type = str,
        help = "Material to be queried with and masked from dataset."
    )
    parser.add_argument(
        "--property",
        type = str,
        choices = property_options,
        help = "Property to predict"
    )
    parser.add_argument(
        "--model",
        type = str,
        help = "LLM Model to use"
    )
    return parser.parse_args()



def main():
    grading = Grading()
    grading
    arguments = get_arguments()
    dataset = arguments.dataset
    material = arguments.material
    chem_property = arguments.property
    model = arguments.model
    main_loop(dataset, material, chem_property, model)


if __name__ == "__main__":
    main()