from api_key import (
    ANTHROPIC_API_KEY,
    GOOGLE_GENAI_API_KEY,
    HUGGINGFACE_API_KEY,
    OPENAI_API_KEY,
)
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from prompts.materials import SYSTEM_MATERIAL
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Literal, Optional, Union

Analogy = Annotated[str, Field(description="The analogy used to arrive at the prediction, including all reasoning steps.")]
Code = Annotated[Optional[str], Field(description="Any Python code used to generate the prediction.")]
Math = Annotated[Optional[str], Field(description="Any LaTeX equations used to generate the prediction.")]


class VolumeDict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: float
    b: float
    c: float
    volume: float

class AllResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analogy: Analogy
    band_gap_prediction: Annotated[float, Field(description="The predicted band gap value; number only, no units.")]
    formation_energy_prediction: Annotated[float, Field(description="The predicted formation energy (per atom); number only, no units.")]
    volume_prediction: Annotated[VolumeDict, Field(description="a, b, c, volume; numbers only, no units.")]

class BandGapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analogy: Analogy
    code: Code = None
    math: Math = None
    band_gap_prediction: Annotated[float, Field(description="The predicted band gap value; number only, no units.")]

class FormationEnergyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analogy: Analogy
    code: Code = None
    math: Math = None
    formation_energy_prediction: Annotated[float, Field(description="The predicted formation energy (per atom); number only, no units.")]

class VolumeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analogy: Analogy
    code: Code = None
    math: Math = None
    volume_prediction: Annotated[VolumeDict, Field(description="a, b, c, volume; numbers only, no units.")]

schema_map = {
    "all": AllResponse,
    "band_gap": BandGapResponse,
    "formation_energy": FormationEnergyResponse,
    "volume": VolumeResponse,
}

# def _system_with_hint(response_type: Choice) -> str:
#     if response_type == "auto":
#         return SYSTEM_MATERIAL
#     return SYSTEM_MATERIAL + (
#         f"\nYou MUST return only the '{response_type}' variant. "
#         f"Set prediction_type='{response_type}' and include only fields of that variant."
#     )

def call_anthropic(prompt: str, response_type: str, model: str = "claude-3-5-sonnet-20241022"):
    try:
        schema = schema_map[response_type]
        llm = init_chat_model(model, model_provider="anthropic", anthropic_api_key=ANTHROPIC_API_KEY)
        messages = [
            SystemMessage(content=SYSTEM_MATERIAL),
            HumanMessage(content=prompt),
        ]
        out = llm.with_structured_output(schema=schema).invoke(messages)
        return out        
    except Exception as e:
        print(e)
        raise

def call_openai(prompt: str, response_type: str, model: str = "gpt-5-mini"):
    schema = schema_map[response_type]
    llm = init_chat_model(model, model_provider="openai", openai_api_key=OPENAI_API_KEY)
    messages = [
        SystemMessage(content=SYSTEM_MATERIAL),
        HumanMessage(content=prompt)
    ]
    try:
        out = llm.with_structured_output(schema=schema).invoke(messages)
        return out
    except Exception as e:
        print(e)
        raise
        # return call_openai(prompt = prompt, response_type = response_type, model = model)


if __name__ == "__main__":
    try:
        print(call_openai("""
Predict the band gap (units: electronvolts), formation energy (units: electronvolts per atom), and lattice parameters/volume (units: angstroms) of:
NdO3

Analogues:
material_id,formula_pretty,is_fit,rms_A,a_A,b_A,c_A,volume_A3,band_gap,formation_energy_per_atom
mp-27971,AcCl3,True,2.2789607510805267e-06,7.695635518178531,7.695635518178531,4.5690029800000005,234.33710146932958,5.0865,-2.788094297499999
mp-22896,LaCl3,True,0.000527085707997413,7.54401242095835,7.54401242095835,4.37168074,215.46853535083147,3.6454,-2.703267628124999
mp-23278,NpCl3,True,0.0006836410439887077,7.421231158358292,7.421231158358292,4.31220639,205.67527616522997,0.047299999999999,-2.3544468428125
mp-23211,PrCl3,True,0.0008459170989922289,7.499680351730479,7.499680351730479,4.30124389,209.51262940784073,4.3454000000000015,-2.592783406041665
mp-23183,NdCl3,True,0.0008874578802475832,7.474769861164252,7.474769861164252,4.24927627,205.60858834099812,4.299799999999999,-2.585905556250001
mp-582011,CeCl3,True,0.0016586754435375234,7.504530635848222,7.504530635848222,4.29801741,209.6263496616267,0.011099999999999001,-2.567102358125
mp-22918,PuCl3,True,0.002148012076400106,7.391154003061179,7.391154003061179,4.25312202,201.21621577358968,0.0,-2.3443700303125
mp-23166,NpBr3,True,0.005058217831420721,8.005590272283271,8.005590272283271,4.41260259000525,244.91318455212155,0.0,-1.8072203021875
mp-23208,UCl3,True,0.006433921926600921,7.503980594290824,4.28613031,7.520283100887413,209.760366052495,0.0,-2.210535621874999
mp-569850,CeBr3,True,0.006562958369122433,8.086764346897269,8.086764346897269,4.39952406,249.1643499160867,0.0,-2.261770134999998
mp-23265,GdCl3,True,0.007179934457747369,7.458317403678345,7.458317403678345,4.07490477,196.30430273272782,2.0054,-2.506313930624998
mp-23255,UBr3,True,0.007238886853524771,8.016692797401575,4.4252762,8.019542978473257,248.12502853657753,0.0,-1.6926844325000001
mp-23221,PrBr3,True,0.0072775818073881685,8.029468645519184,8.029468645519184,4.43424121,247.5845606332728,3.429599999999999,-2.306216135416666
mp-23263,LaBr3,True,0.007876390302554988,8.05395551657293,8.05395551657293,4.51034726,253.37225998432237,2.9266,-2.423042803749999
mp-27972,AcBr3,True,0.008308517942889792,8.168264033291733,8.168264033291733,4.72345121,272.9289475926483,4.103299999999999,-2.494519163125
mp-569895,EuCl3,True,0.008427771560605615,7.435378276232042,4.223114890000122,7.485639604170291,204.5083633107856,0.0,-2.125660104375001
mp-568170,TbCl3,True,0.00936562003871722,7.43151403067669,7.43151403067669,4.01996813,192.2683706224788,4.1024,-2.532088784374998
mp-642855,Mg3Ge,True,0.03598575872173,6.952656121341288,6.952656121341288,4.252652,178.02950174529641,0.0,-0.11605399375000101
mp-1206505,GdO3,True,0.04241344351919332,6.084385635216949,6.084385635216949,3.610265,115.7452500329357,0.0,-1.766396052499999
mp-1205900,CeO3,True,0.043943639467043286,6.0648841667073885,6.0648841667073885,3.64122,115.99054060080472,0.0,-2.434195327499999
mp-720472,PrO3,True,0.10464192904979736,5.6299501203618565,5.6299501203618565,4.41088,121.07792478535103,0.6315000000000001,-1.938387684166666
mp-1178953,TmO3,True,0.11595811742071667,6.393507849615106,6.393507849615106,3.360292,118.95591862598927,0.20350000000000001,-2.094708850000001
mp-1206099,YO3,True,0.1163996601504484,6.4975282988650696,6.4975282988650696,3.423774,125.1791737455225,0.194799999999999,-2.0104380375
mp-1025483,ErO3,True,0.11678329909529864,6.442921512167191,6.442921512167191,3.372533,121.2418406885253,0.20980000000000001,-2.077117393749999
mp-1025565,HoO3,True,0.12023129203274523,6.487932030299639,6.487932030299639,3.40243,124.03161925345216,0.1889,-2.062738198749999
mp-1025387,TbO3,True,0.12317161730994096,6.553241978718663,6.553241978718663,3.447799,128.22862349666522,0.156399999999999,-2.045014071666666
mp-1025421,SmO3,True,0.1306855109493941,6.717208532387373,6.717208532387373,3.51995,137.5449938078976,0.15380000000000002,-1.9860494700000002
    """, "all"))
    except Exception as e:
        print(f"[ERROR] - inference : {e}")