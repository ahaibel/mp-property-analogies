# Materials Project Property Analogies
---------------------
## Setup

Create a virtual environment (venv, conda, uv, your choice).
The dependency list (requirements.txt) will be updated as project proceeds.

```
conda create --name mp-property-analogies python=3.13
conda activate mp-property-analogies
pip install -r requirements.txt
```

Register for the [Materials Project](https://next-gen.materialsproject.org/) (I recommend using school account as Google login).
Navigate to the [API page](https://next-gen.materialsproject.org/api) to see your API Key.
Create an `api_key.py` file in the root directory with the following content:

```
ANTHROPIC_API_KEY=""
HUGGINGFACE_API_KEY=""
GOOGLE_GENAI_API_KEY=""
OPENAI_API_KEY=""
MATERIALS_PROJECT_API_KEY=""
```

Add your respective API keys in the quotes.

## Data Generation

The `mp_structural_analogues.py` script takes in a material id (mp-x) from the [Materials Project Explorer](https://next-gen.materialsproject.org/materials). It outputs as a .csv materials with structural similarity and a few properties (currently band gap, formation energy, and lattice volume). This list is sorted by structural similarity; the last entries (2~30) should be manually checked. Visually compare the structures in the Materials Project Explorer and remove ones that don't match. If in doubt, remove it.

Scents Data came from Keller & Vosshall 2016. See (see [olfactory_analogical_reasoning](https://github.com/ahaibel/mp-property-analogies/tree/olfactory_analogical_reasoning) branch)

## Usage
Example usage:

```
python main.py --dataset 176_AB3_mp-27971.csv --material PrBr3 --property volume --model gpt-5-mini
```

Arguments:
- dataset: A `.csv` file from /datasets
- material: A material formula from that `.csv` file (second column)
- property: The property to predict (options: band_gap, formation_energy, volume)
- model: Model name to be used (only gpt-5-mini tested, code for other providers incomplete)

## Sample Results
NdClO predictions from dataset [129_ABC_mp-30273.csv](https://github.com/ahaibel/mp-property-analogies/blob/main/datasets/129_ABC_mp-30273.csv). The later trials have successively reduced support to draw analogies from, with no elements from the test material found in the analogy support provided to the LLM.

image here

Scent predictions from [keller_molecules_merged.csv](https://github.com/ahaibel/mp-property-analogies/blob/olfactory_analogical_reasoning/keller_molecules_merged.csv). Predictions (0-100 scale for each category) were significantly weaker here than on Materials Project data, and with no perturbation.

image here

## Incomplete
- All model provider support
- Output format (needs data permutation descriptions)
- Quantitative Grading (MAE, perhaps other metrics)
- Qualitative Grading - analogy quality, categorization
- Scents (see [olfactory_analogical_reasoning](https://github.com/ahaibel/mp-property-analogies/tree/olfactory_analogical_reasoning) branch)