from api_key import (
    ANTHROPIC_API_KEY,
    GOOGLE_GENAI_API_KEY,
    HUGGINGFACE_API_KEY,
    OPENAI_API_KEY,
)
from openai import OpenAI
from prompts.scents import SYSTEM_SCENT, USER_SCENT
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from string import Template
import json
import pandas as pd
from tqdm import tqdm


client = OpenAI(api_key = OPENAI_API_KEY)
HundredScale = Annotated[float, Field(ge=0.0, le=100.0)]


class ScentResponseTwentyTwo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # explanation: Annotated[str, Field(description="Justification for the given predictions.")]
    analogy: Annotated[str, Field(description="The analogy used to arrive at the predictions, including all reasoning steps.")]
    edible: Annotated[HundredScale, Field(description="Predicted rating for 'edible' aspect.")]
    bakery: Annotated[HundredScale, Field(description="Predicted rating for 'bakery' aspect.")]
    sweet: Annotated[HundredScale, Field(description="Predicted rating for 'sweet' aspect.")]
    fruit: Annotated[HundredScale, Field(description="Predicted rating for 'fruit' aspect.")]
    fish: Annotated[HundredScale, Field(description="Predicted rating for 'fish' aspect.")]
    garlic: Annotated[HundredScale, Field(description="Predicted rating for 'garlic' aspect.")]
    spices: Annotated[HundredScale, Field(description="Predicted rating for 'spices' aspect.")]
    cold: Annotated[HundredScale, Field(description="Predicted rating for 'cold' aspect.")]
    sour: Annotated[HundredScale, Field(description="Predicted rating for 'sour' aspect.")]
    burnt: Annotated[HundredScale, Field(description="Predicted rating for 'burnt' aspect.")]
    acid: Annotated[HundredScale, Field(description="Predicted rating for 'acid' aspect.")]
    warm: Annotated[HundredScale, Field(description="Predicted rating for 'warm' aspect.")]
    musky: Annotated[HundredScale, Field(description="Predicted rating for 'musky' aspect.")]
    sweaty: Annotated[HundredScale, Field(description="Predicted rating for 'sweaty' aspect.")]
    ammonia: Annotated[HundredScale, Field(description="Predicted rating for 'ammonia' aspect.")]
    decayed: Annotated[HundredScale, Field(description="Predicted rating for 'decayed' aspect.")]
    wood: Annotated[HundredScale, Field(description="Predicted rating for 'wood' aspect.")]
    grass: Annotated[HundredScale, Field(description="Predicted rating for 'grass' aspect.")]
    flower: Annotated[HundredScale, Field(description="Predicted rating for 'flower' aspect.")]
    chemical: Annotated[HundredScale, Field(description="Predicted rating for 'chemical' aspect.")]
    strength: Annotated[HundredScale, Field(description="Predicted rating for 'strength' aspect.")]
    pleasant: Annotated[HundredScale, Field(description="Predicted rating for 'pleasant' aspect.")]
    familiar: Annotated[HundredScale, Field(description="Predicted rating for 'familiar' aspect.")]


class ScentResponseSeven(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # explanation: Annotated[str, Field(description="Justification for the given predictions.")]
    analogy: Annotated[str, Field(description="The analogy used to arrive at the predictions, including all reasoning steps.")]
    fish: Annotated[HundredScale, Field(description="Predicted rating for 'fish' aspect.")]
    cold: Annotated[HundredScale, Field(description="Predicted rating for 'cold' aspect.")]
    ammonia: Annotated[HundredScale, Field(description="Predicted rating for 'ammonia' aspect.")]
    decayed: Annotated[HundredScale, Field(description="Predicted rating for 'decayed' aspect.")]
    strength: Annotated[HundredScale, Field(description="Predicted rating for 'strength' aspect.")]
    pleasant: Annotated[HundredScale, Field(description="Predicted rating for 'pleasant' aspect.")]
    familiar: Annotated[HundredScale, Field(description="Predicted rating for 'familiar' aspect.")]


def load_analogues_data(csv_file, target_molecule):
    """Load CSV and format for prompts, hiding target molecule"""
    df = pd.read_csv(csv_file)
    df_filtered = df[df['OdorName'] != target_molecule]
    print(f"Target '{target_molecule}' hidden. Using {len(df_filtered)} analogues out of {len(df)} total molecules.")
    return df_filtered.to_csv()


def call_openai(user_prompt):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "scent_response",
            "description": "Respond as a JSON that strictly matches the schema.",
            "schema": ScentResponseTwentyTwo.model_json_schema(),
            "strict": True,
        }
    }
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=response_format,
            messages=[
                {"role": "system", "content": SYSTEM_SCENT},
                {"role": "user", "content": user_prompt},
            ],
        )
        # return response.choices[0].message.content
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return f"ERROR: API call failed - {str(e)}"


def predict_one_molecule(csv_file, query_molecule):
    """Predict scent ratings for one molecule"""
    analogues_data = load_analogues_data(csv_file, query_molecule)
    user_prompt = Template(USER_SCENT).substitute(molecule = query_molecule, df = analogues_data)
    prediction = call_openai(user_prompt)
    # prediction["molecule"] = query_molecule
    return prediction


# def calculate_prediction_errors(predicted, actual):
#     """Calculate prediction errors for available dimensions"""
#     if predicted == "PARSE_ERROR" or not isinstance(predicted, dict):
#         return {"error": "Could not parse prediction"}
    
#     errors = {}
#     all_rating_cols = [
#         "EDIBLE", "BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES",
#         "COLD", "SOUR", "BURNT", "ACID", "WARM", "MUSKY", "SWEATY",
#         "AMMONIA/URINOUS", "DECAYED", "WOOD", "GRASS", "FLOWER", "CHEMICAL",
#         "HOW STRONG IS THE SMELL?", "HOW PLEASANT IS THE SMELL?", 
#         "HOW FAMILIAR IS THE SMELL?"
#     ]
    
#     # Only calculates errors for dimensions that exist in both prediction and ground truth
#     for col in all_rating_cols:
#         if col in predicted and col in actual:
#             try:
#                 pred_val = float(predicted[col])
#                 actual_val = float(actual[col])
#                 errors[f"{col}_error"] = abs(pred_val - actual_val)
#                 errors[f"{col}_predicted"] = pred_val
#                 errors[f"{col}_actual"] = actual_val
#             except (ValueError, TypeError):
#                 errors[f"{col}_error"] = "PARSE_ERROR"
    
#     valid_errors = [v for k, v in errors.items() if k.endswith('_error') and v != "PARSE_ERROR"]
#     if valid_errors:
#         errors["mean_absolute_error"] = sum(valid_errors) / len(valid_errors)
#         errors["dimensions_evaluated"] = len(valid_errors)
    
#     return errors


def run_experiment(csv_file, n_molecules=20):
    """Run the molecular analogical reasoning experiment"""
    df = pd.read_csv(csv_file)
    df = df.head(n_molecules)
    print(f"Testing {len(df)} molecules using basic prompting")
    results = []
    
    for i, row in tqdm(df.iterrows(), desc = "Querying molecules..."):
        query_molecule = row['OdorName']
        scores = predict_one_molecule(csv_file, query_molecule)
        result = {"molecule": query_molecule} | scores
        
        # result.update({
        #     "GT_FISH": row['FISH'],
        #     "GT_COLD": row['COLD'],
        #     "GT_AMMONIA": row['AMMONIA/URINOUS'],
        #     "GT_DECAYED": row['DECAYED'],
        #     "GT_STRENGTH": row['HOW STRONG IS THE SMELL?'],
        #     "GT_PLEASANT": row['HOW PLEASANT IS THE SMELL?'],
        #     "GT_FAMILIAR": row['HOW FAMILIAR IS THE SMELL?']
        # })
        
        # if 'ExtractedRatings' in result:
        #     error_metrics = calculate_prediction_errors(result['ExtractedRatings'], row)
        #     result.update(error_metrics)
        
        results.append(result)
        # print(f"Completed {i+1}/{len(df)}: {query_molecule}")
    
    #results
    output_filename = f"scent_results_forced_analogy.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
    
 
    # mae_values = results_df['mean_absolute_error'].dropna()
    # if len(mae_values) > 0:
    #     print(f"Mean Absolute Error: {mae_values.mean():.2f} ± {mae_values.std():.2f}")
    #     print(f"Range: {mae_values.min():.2f} - {mae_values.max():.2f}")
    
    return results

if __name__ == "__main__":
    results = run_experiment(
        csv_file="datasets/keller_molecules_merged.csv",
        n_molecules=100
    )
    
    print(f"\nCompleted {len(results)} predictions")
    print("Ready for visualization and analysis!")

#     prediction_test = """
# Predict the scent perception ratings (0-100, decimal acceptable) for this molecule:
# 2-Phenylethyl isothiocyanate

# Here are structurally related molecules with their scent perception ratings:
# 0,CID,OdorName,CAS,CanonicalSMILES,MolecularWeight,fish,cold,ammonia,decayed,strength,pleasant,familiar
# 2,17121,1-Octen-3-yl acetate,2442-10-6,CCCCCC(C=C)OC(=O)C,170.25,14.5,35.09090909090909,15.818181818181818,20.5,44.14457831325301,37.59036144578313,34.25301204819277
# 3,520191,"3-Acetyl-2,5-dimethylthiophene",2530-10-1,CC1=CC(=C(S1)C)C(=O)C,154.23,42.8,25.11111111111111,26.90909090909091,25.25,56.15730337078652,35.96629213483146,42.79775280898876
# 4,92979,Androstadienone,4075-07-4,CC12CCC3C(C1CC=C2)CCC4=CC(=O)CCC34C,270.4,6.0,21.352941176470587,29.956521739130437,27.416666666666668,52.16,33.333333333333336,39.52
# 5,62580,Piperonyl isobutyrate,5461-08-5,CC(C)C(=O)OCC1=CC2=C(C=C1)OCO2,222.24,5.0,27.0,21.142857142857142,25.5,27.894736842105264,45.719298245614034,34.40350877192982
# 6,24197,Diethyl malate,7554-12-3,CCOC(=O)CC(C(=O)OCC)O,190.19,41.833333333333336,30.727272727272727,19.9,18.857142857142858,19.96153846153846,46.75,30.48076923076923
# 7,7476,Acetanisole,100-06-1,CC(=O)C1=CC=C(C=C1)OC,150.17,9.666666666666666,26.46153846153846,32.625,17.333333333333332,64.48543689320388,60.53398058252427,53.45631067961165
# 8,61252,"4,5-Dihydro-3(2H)-thiophenone",1003-04-9,C1CSCC1=O,102.16,38.54545454545455,16.5,28.8,29.291666666666668,69.33734939759036,25.397590361445783,45.43373493975904
# 9,24834,4-Ethoxybenzaldehyde,10031-82-0,CCOC1=CC=C(C=C1)C=O,150.17,1.0,28.714285714285715,19.857142857142858,11.666666666666666,51.723404255319146,66.94680851063829,53.54255319148936
# 10,7500,ethylbenzene,100-41-4,CCC1=CC=CC=C1,106.16,1.0,28.0,23.5,31.625,59.1125,39.9625,46.9
# 11,244,benzyl alcohol,100-51-6,C1=CC=C(C=C1)CO,108.14,4.333333333333333,35.473684210526315,29.5625,19.0,60.47524752475248,49.3960396039604,48.0
# 12,240,benzaldehyde,100-52-7,C1=CC=C(C=C1)C=O,106.12,10.5,20.5,15.7,16.6,60.54651162790697,62.63953488372093,55.53488372093023
# 13,7519,anisole,100-66-3,COC1=CC=CC=C1,108.14,22.6,28.33333333333333,28.33333333333333,21.0,59.795180722891565,36.6144578313253,41.65060240963855
# 14,7559,Methyl phenylacetate,101-41-7,COC(=O)CC1=CC=CC=C1,150.17,9.0,39.41176470588236,34.5,27.894736842105264,63.96153846153846,42.60576923076923,51.89423076923077
# 15,7583,Diphenyl ether,101-84-8,C1=CC=C(C=C1)OC2=CC=CC=C2,170.21,11.5,31.181818181818183,35.625,5.0,46.86666666666667,48.906666666666666,40.973333333333336
# 16,7593,"1,3-Diphenyl-2-propanone",102-04-5,C1=CC=C(C=C1)CC(=O)CC2=CC=CC=C2,210.27,14.333333333333334,26.2,27.310344827586206,20.2,59.694736842105264,37.14736842105263,49.02105263157895
# 17,60998,Isobutyl phenylacetate,102-13-6,CC(C)COC(=O)CC1=CC=CC=C1,192.25,18.666666666666668,23.416666666666668,23.55555555555556,25.785714285714285,39.60526315789474,48.53947368421053,43.76315789473684
# 18,60999,Benzyl phenylacetate,102-16-9,C1=CC=C(C=C1)CC(=O)OCC2=CC=CC=C2,226.27,33.0,25.4,26.625,22.727272727272727,48.36986301369863,36.71232876712329,45.36986301369863
# 19,7601,Phenethyl phenylacetate,102-20-5,C1=CC=C(C=C1)CCOC(=O)CC2=CC=CC=C2,240.3,23.666666666666668,31.0,22.857142857142858,26.25,34.11666666666667,53.18333333333333,32.36666666666667
# 20,5541,Triacetin,102-76-1,CC(=O)OCC(COC(=O)C)OC(=O)C,218.2,14.75,21.307692307692307,25.428571428571427,28.22222222222222,51.3956043956044,36.03296703296704,46.75824175824176
#     """

#     print(call_openai(prediction_test))