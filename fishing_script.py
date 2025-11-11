#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd
import re
from openai import OpenAI


import os
os.environ['OPENAI_API_KEY'] = "provide your key here"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OPENAI_CONFIG = {
    "temperature": 0,
    "model": "gpt-4o-mini"
}

SYSTEM_PROMPT = """You are an expert in chemistry, molecular structure, and olfactory science. You understand relationships between molecular structure and scent perception."""

def basic_prompt(analogues_data, query_molecule):
    return f"""
Here are structurally related molecules with their scent perception ratings:
{analogues_data}

What are the scent perception ratings for {query_molecule}?
Use analogical reasoning to predict based on molecular similarities and scent patterns.

Format the final answer like this - 
<scent_ratings>
{{
    "molecule": "{query_molecule}",
    "EDIBLE": your_prediction_for_edible_rating,
    "BAKERY": your_prediction_for_bakery_rating,
    "SWEET": your_prediction_for_sweet_rating,
    "FRUIT": your_prediction_for_fruit_rating,
    "FISH": your_prediction_for_fish_rating,
    "GARLIC": your_prediction_for_garlic_rating,
    "SPICES": your_prediction_for_spices_rating,
    "COLD": your_prediction_for_cold_rating,
    "SOUR": your_prediction_for_sour_rating,
    "BURNT": your_prediction_for_burnt_rating,
    "ACID": your_prediction_for_acid_rating,
    "WARM": your_prediction_for_warm_rating,
    "MUSKY": your_prediction_for_musky_rating,
    "SWEATY": your_prediction_for_sweaty_rating,
    "AMMONIA/URINOUS": your_prediction_for_ammonia_rating,
    "DECAYED": your_prediction_for_decayed_rating,
    "WOOD": your_prediction_for_wood_rating,
    "GRASS": your_prediction_for_grass_rating,
    "FLOWER": your_prediction_for_flower_rating,
    "CHEMICAL": your_prediction_for_chemical_rating,
    "HOW STRONG IS THE SMELL?": your_prediction_for_intensity,
    "HOW PLEASANT IS THE SMELL?": your_prediction_for_pleasantness,
    "HOW FAMILIAR IS THE SMELL?": your_prediction_for_familiarity
}}
</scent_ratings>

The prediction should only contain the JSON object and no other text.
    """

def load_analogues_data(csv_file, target_molecule):
    """Load CSV and format for prompts, hiding target molecule"""
    df = pd.read_csv(csv_file)
    df_filtered = df[df['OdorName'] != target_molecule]
    
    print(f"Target '{target_molecule}' hidden. Using {len(df_filtered)} analogues out of {len(df)} total molecules.")
    
    analogues_text = ""
    for _, row in df_filtered.iterrows():
        analogues_text += f"Molecule: {row['OdorName']}\n"
        analogues_text += f"Ratings - FISH: {row['FISH']:.1f}, COLD: {row['COLD']:.1f}, "
        analogues_text += f"AMMONIA/URINOUS: {row['AMMONIA/URINOUS']:.1f}, DECAYED: {row['DECAYED']:.1f}, "
        analogues_text += f"STRENGTH: {row['HOW STRONG IS THE SMELL?']:.1f}, "
        analogues_text += f"PLEASANTNESS: {row['HOW PLEASANT IS THE SMELL?']:.1f}, "
        analogues_text += f"FAMILIARITY: {row['HOW FAMILIAR IS THE SMELL?']:.1f}\n\n"
    
    return analogues_text.strip()

def extract_scent_ratings(raw_output):
    """Extract scent ratings from LLM predictions"""
    try:
        output = raw_output.strip()
        match = re.search(r"<scent_ratings>\s*(.*?)\s*</scent_ratings>", output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return "PARSE_ERROR"
        return "PARSE_ERROR"
    except Exception:
        return "PARSE_ERROR"

def call_openai(prompt):
    """Call OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_CONFIG["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=OPENAI_CONFIG["temperature"]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: API call failed - {str(e)}"

def predict_one_molecule(csv_file, query_molecule):
    """Predict scent ratings for one molecule"""
    analogues_data = load_analogues_data(csv_file, query_molecule)
    prompt = basic_prompt(analogues_data, query_molecule)
    prediction = call_openai(prompt)
    
    return {
        "QueryMolecule": query_molecule,
        "PromptType": "basic",
        "Prediction": prediction,
        "ExtractedRatings": extract_scent_ratings(prediction),
    }

def calculate_prediction_errors(predicted, actual):
    """Calculate prediction errors for available dimensions"""
    if predicted == "PARSE_ERROR" or not isinstance(predicted, dict):
        return {"error": "Could not parse prediction"}
    
    errors = {}
    all_rating_cols = [
        "EDIBLE", "BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES",
        "COLD", "SOUR", "BURNT", "ACID", "WARM", "MUSKY", "SWEATY",
        "AMMONIA/URINOUS", "DECAYED", "WOOD", "GRASS", "FLOWER", "CHEMICAL",
        "HOW STRONG IS THE SMELL?", "HOW PLEASANT IS THE SMELL?", 
        "HOW FAMILIAR IS THE SMELL?"
    ]
    
    # Only calculates errors for dimensions that exist in both prediction and ground truth
    for col in all_rating_cols:
        if col in predicted and col in actual:
            try:
                pred_val = float(predicted[col])
                actual_val = float(actual[col])
                errors[f"{col}_error"] = abs(pred_val - actual_val)
                errors[f"{col}_predicted"] = pred_val
                errors[f"{col}_actual"] = actual_val
            except (ValueError, TypeError):
                errors[f"{col}_error"] = "PARSE_ERROR"
    
    valid_errors = [v for k, v in errors.items() if k.endswith('_error') and v != "PARSE_ERROR"]
    if valid_errors:
        errors["mean_absolute_error"] = sum(valid_errors) / len(valid_errors)
        errors["dimensions_evaluated"] = len(valid_errors)
    
    return errors

def run_experiment(csv_file, n_molecules=20):
    """Run the molecular analogical reasoning experiment"""
    df = pd.read_csv(csv_file)
    df = df.head(n_molecules)
    
    print(f"Testing {len(df)} molecules using basic prompting")
    
    results = []
    
    for i, row in df.iterrows():
        query_molecule = row['OdorName']
        
        result = predict_one_molecule(csv_file, query_molecule)
        
        result.update({
            "GT_FISH": row['FISH'],
            "GT_COLD": row['COLD'],
            "GT_AMMONIA": row['AMMONIA/URINOUS'],
            "GT_DECAYED": row['DECAYED'],
            "GT_STRENGTH": row['HOW STRONG IS THE SMELL?'],
            "GT_PLEASANT": row['HOW PLEASANT IS THE SMELL?'],
            "GT_FAMILIAR": row['HOW FAMILIAR IS THE SMELL?']
        })
        
        if 'ExtractedRatings' in result:
            error_metrics = calculate_prediction_errors(result['ExtractedRatings'], row)
            result.update(error_metrics)
        
        results.append(result)
        print(f"Completed {i+1}/{len(df)}: {query_molecule}")
    
    #results
    output_filename = f"molecular_analogy_results_basic_{OPENAI_CONFIG['model'].replace('-', '_')}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
    
 
    mae_values = results_df['mean_absolute_error'].dropna()
    if len(mae_values) > 0:
        print(f"Mean Absolute Error: {mae_values.mean():.2f} ± {mae_values.std():.2f}")
        print(f"Range: {mae_values.min():.2f} - {mae_values.max():.2f}")
    
    return results

if __name__ == "__main__":
    results = run_experiment(
        csv_file="keller_molecules_merged.csv",
        n_molecules=100
    )
    
    print(f"\nCompleted {len(results)} predictions")
    print("Ready for visualization and analysis!")

