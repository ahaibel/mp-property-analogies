import argparse
import itertools as it
import pandas as pd
from parse_and_prompt import main_loop
from pymatgen.core.composition import Composition

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
    arguments = get_arguments()
    dataset = arguments.dataset
    material = arguments.material
    chem_property = arguments.property
    model = arguments.model
    main_loop(dataset, material, chem_property, model)


if __name__ == "__main__":
    main()