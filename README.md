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

## Usage

The `mp_structural_analogues.py` script takes in a material id (mp-x) from the [Materials Project Explorer](https://next-gen.materialsproject.org/materials). It outputs as a .csv materials with structural similarity and a few properties (currently band gap, formation energy, and lattice volume). This list is sorted by structural similarity; the last entries (2~30) should be manually checked. Visually compare the structures in the Materials Project Explorer and remove ones that don't match. If in doubt, remove it.


## Running
Example usage

```
python main.py --dataset 176_AB3_mp-27971.csv --material PrBr3 --property volume --model gpt-5-mini
```

Works in Progress:
- Cleanup on mp_structural_analogues.py
- Element Filtering in Prompt Data
- Input Prompt Generation
- LLM API interfacing
- LLM Output Filtering (Answer)
- Grading (Quantitative)
- Grading (Qualitative)