import llm_analogies
from string import Template
from prompts.materials import(
    USER_BAND_GAP,
    USER_FORMATION_ENERGY,
    USER_VOLUME,
)

MODEL_FAMILIES = {
    "anthropic": {
        "claude-3-7-sonnet-20250219",
    },
    "google_genai": {
        "gemini-2.5-flash",
    },
    "huggingface": {
        "Qwen/Qwen3-Next-80B-A3B-Thinking",
    },
    "openai": {
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    }
}


def get_model_family(model_name):
    return next(
        (family for family, models in MODEL_FAMILIES.items() if model_name in models),
        "Invalid model name",
    )


def run_inference(df, material, response_type, model):
    if response_type == "band_gap":
        prompt = Template(USER_BAND_GAP).substitute(
            material = material,
            df = df.to_csv(index=False)
        )
    elif response_type == "formation_energy":
        prompt = Template(USER_FORMATION_ENERGY).substitute(
            material = material,
            df = df.to_csv(index=False)
        )
    elif response_type == "volume":
        prompt = Template(USER_VOLUME).substitute(
            material = material,
            df = df.to_csv(index=False)
        )

    model_family = get_model_family(model)
    if model_family == "anthropic":
        return llm_analogies.call_anthropic(prompt, response_type, model)
    elif model_family == "google_genai":
        return llm_analogies.call_google_genai(prompt, response_type, model)
    elif model_family == "huggingface":
        return llm_analogies.call_huggingface(prompt, response_type, model)
    elif model_family == "openai":
        return llm_analogies.call_openai(prompt, response_type, model)


if __name__ == "__main__":
    print(get_model_family("gpt-5"))