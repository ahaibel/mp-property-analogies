from api_key import ANTHROPIC_API_KEY, GOOGLE_GENAI_API_KEY, HUGGINGFACE_API_KEY, OPENAI_API_KEY
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, Union

SYSTEM_PROMPT = (
    "You are tasked with a finding a property for a given crystal structure, and are "
    "given a list of analogous crystal structures with their data. Construct an analogy "
    "of the form A is to B as C is to D in order to find the property of D (the query)."
)

Analogy = Annotated[str, Field(description="The analogy used to arrive at the prediction, including all reasoning steps.")]
Code = Annotated[Optional[str], Field(description="Any Python code used to generate the prediction.")]
Math = Annotated[Optional[str], Field(description="Any LaTeX equations used to generate the prediction.")]

class BandGapResponse(BaseModel):
    prediction_type: Literal["band_gap"] = "band_gap"
    analogy: Analogy
    code: Code = None
    math: Math = None
    band_gap_prediction: Annotated[float, Field(description="The predicted band gap value; number only, no units.")]

class FormationEnergyResponse(BaseModel):
    prediction_type: Literal["formation_energy"] = "formation_energy"
    analogy: Analogy
    code: Code = None
    math: Math = None
    formation_energy_prediction: Annotated[float, Field(description="The predicted formation energy (per atom); number only, no units.")]

class VolumeDict(BaseModel):
    a: float
    b: float
    c: float
    volume: float

class VolumeResponse(BaseModel):
    prediction_type: Literal["volume"] = "volume"
    analogy: Analogy
    code: Code = None
    math: Math = None
    volume_prediction: Annotated[VolumeDict, Field(description="a, b, c, volume; numbers only, no units.")]

Choice = Literal["auto", "band_gap", "formation_energy", "volume"]
Prediction = Annotated[
    Union[BandGapResponse, FormationEnergyResponse, VolumeResponse],
    Field(discriminator="prediction_type")
]

schema_map = {
    "band_gap": BandGapResponse,
    "formation_energy": FormationEnergyResponse,
    "volume": VolumeResponse,
    "auto": Prediction,
}

def _system_with_hint(response_type: Choice) -> str:
    if response_type == "auto":
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + (
        f"\nYou MUST return only the '{response_type}' variant. "
        f"Set prediction_type='{response_type}' and include only fields of that variant."
    )

def call_anthropic(prompt: str, response_type: Choice, model: str = "claude-3-5-sonnet-20241022"):
    try:
        schema = schema_map[response_type]
        llm = init_chat_model(model, model_provider="anthropic", anthropic_api_key=ANTHROPIC_API_KEY)
        messages = [
            SystemMessage(content=_system_with_hint(response_type)),
            HumanMessage(content=prompt),
        ]
        out = llm.with_structured_output(schema=schema).invoke(messages)
        if response_type != "auto" and getattr(out, "prediction_type", None) != response_type:
            raise ValueError(f"Expected prediction_type='{response_type}', got '{getattr(out, 'prediction_type', None)}'")
        return out        
    except Exception as e:
        print("[ERROR] - claude function:\n", str(e))
        raise

def call_openai(prompt: str, response_type: Choice, model: str = "gpt-5-mini"):
    schema = schema_map[response_type]
    llm = init_chat_model(model, model_provider="openai", openai_api_key=OPENAI_API_KEY)
    messages = [
        SystemMessage(content=_system_with_hint(response_type)),
        HumanMessage(content=prompt),
    ]
    out = llm.with_structured_output(schema=schema).invoke(messages)
    # Optional hard guard when not auto:
    if response_type != "auto" and getattr(out, "prediction_type", None) != response_type:
        raise ValueError(f"Expected prediction_type='{response_type}', got '{getattr(out, 'prediction_type', None)}'")
    return out


if __name__ == "__main__":
    try:
        print(call_openai("""
What are the lattice parameters and volume of EuAl3Cu2?
The numbers in this list are in units of angstroms.

formula_pretty,a_A,b_A,c_A,volume_A3
DyB2Ru3,5.472512401,5.472512401,3.02102166,78.35342399
CeGa3Pd2,5.379950081,5.379950081,4.27102198,107.0579518
UB2Ir3,5.428118572,5.428118572,3.19647148,81.56430072
LuB2Rh3,5.403814346,5.403814346,3.118804,78.87140078
GdB2Ir3,5.457449976,5.457449976,3.14849292,81.21063006
LaCu2Ni3,5.041361768,5.041361768,3.92775182,86.45107487
SrAg3Pd2,5.548575314,5.548575314,4.55615608,121.4764791
DyCo3B2,5.022953732,5.022953732,2.92844765,63.98621958
ScCo3B2,4.868916666,4.868916666,2.93481571,60.2526496
SmCo2Ni3,4.907883471,4.907883471,3.966268,82.73726735
EuB2Rh3,5.611641342,5.611641342,2.9369823,80.09617431
YbB2Ir3,5.407580164,5.407580164,3.138913493,79.49062493
LaGe2Ir3,5.694173854,5.694173854,3.855797,108.2695268
NdGa2Pt3,5.584433126,5.584433126,3.99424907,107.8757837
ThMn2Co3,5.112965807,5.112965807,3.904109,88.38902492
ErB2Ru3,5.457099578,5.457099578,3.01629707,77.79084717
CeCo3B2,5.118881773,5.118881773,2.89148443,65.61478165
SmCo3B2,5.089332747,5.089332747,2.93169067,65.76131206
UGa2Cu3,5.239928103,5.239928103,4.023449,95.67088437
TbB2Ru3,5.481132604,5.481132604,3.02242509,78.6369744
TbB2Rh3,5.379841398,5.379841398,3.14173113,78.74782945
YbB2Rh3,5.337565229,5.337565229,3.136095607,77.37600748
LuB2Os3,5.519705733,5.519705733,3.031391,79.98424302
PrAl3Ni2,5.333836092,5.333836092,4.00831703,98.75794107
GdB2Ru3,5.49880754,5.49880754,3.0244028,79.19673948
PrCo3Ni2,4.975994341,4.975994341,4.014188,86.07719289
ErCo3B2,4.951747255,4.951747255,2.96784629,63.02153472
LaSi2Ru3,5.799268255,5.799268255,3.49226,101.7146708
ThAl3Ni2,5.291074509,5.291074509,4.04189658,97.99492465
YbGa2Cu3,5.178751393,5.178751393,3.926303211,91.19364888
ErSi2Rh3,5.435096109,5.435096109,3.60686629,92.27310415
YbAl2Cu3,5.187983345,5.187983345,3.931796645,91.64712082
CeB2Ru3,5.518839469,5.518839469,2.99508313,79.00144514
DyGa2Cu3,5.254933853,5.254933853,4.015535,96.0303595
HoB2Ir3,5.423128569,5.423128569,3.14916601,80.20953307
GdSi2Rh3,5.512994759,5.512994759,3.648934022,96.04434556
DySi2Rh3,5.48342671,5.48342671,3.64914379,95.02233422
LuB2Ru3,5.438786833,5.438786833,3.01059029,77.1234348
TbCo3B2,5.036894085,5.036894085,2.928657,64.34647784
ThFe2Ni3,4.985604122,4.985604122,3.998749,86.07764125
CeZn3Pd2,5.308993193,5.308993193,4.3190311,105.4244393
ThB2Ru3,5.521006487,5.521006487,3.07613235,81.20301669
YbB2Ru3,5.443656321,5.443656321,3.013138133,77.32698385
DyIn3Cu2,5.824156411,5.824156411,4.257596,125.0723011
YSi2Rh3,5.471937707,5.471937707,3.60373789,93.44716217
LuCo3B2,4.917641388,4.917641388,2.96773699,62.15409568
PuB2Ir3,5.479551899,5.479551899,3.124587,81.2481241
SmB2Ru3,5.514365457,5.514365457,3.0274644,79.72614527
GdAl3Pd2,5.397874552,5.397874552,4.19298332,105.803331
PrGa2Co3,5.358789882,5.358789882,3.737016,92.93708494
PrB2Rh3,5.469149906,5.469149906,3.13998285,81.3387628
SmSi2Rh3,5.533836233,5.533836233,3.64973037,96.7930151
GdB2Rh3,5.398516235,5.398516235,3.14106055,79.27856192
PrB2Ir3,5.674472584,5.674472584,2.96773018,82.75724967
UAl3Pd2,5.38423045,5.38423045,4.239652,106.4407979
SmCo3Ni2,4.923375894,4.923375894,4.040514,84.81901748
ErCo3Ni2,4.812887882,4.812887882,4.02729,80.78951944
LaAl2Pt3,5.636957973,5.636957973,3.95265283,108.7699419
LaCu3Ni2,5.010197653,5.010197653,4.04750259,87.98881836
PrAl2Ni3,5.321853048,5.321853048,3.76175416,92.26708458
SmCo2Cu3,4.946805713,4.946805713,4.11776,87.2652565
PuB2Rh3,5.620546241,5.620546241,2.896267296,79.23668948
LaAl3Pd2,5.513867927,5.513867927,4.23531601,111.5139425
YCo3Ni2,4.863590219,4.863590219,4.007545,82.09618807
NdAl2Pt3,5.58176655,5.58176655,3.9563234,106.7494765
LaB2Ru3,5.581024736,5.581024736,3.02334573,81.55420144
GdCo3B2,5.059458551,5.059458551,2.93256239,65.01086983
LaGa2Pt3,5.627870758,5.627870758,4.00590355,109.8801772
LaSi2Rh3,5.620108013,5.620108013,3.64530382,99.71343723
YCo3B2,5.032871953,5.032871953,2.92939292,64.25989648
CeZn2Ni3,5.148842127,5.148842127,3.86938259,88.83650342
CeAl3Pd2,5.432649333,5.432649333,4.22537958,107.9989932
PuAl3Pd2,5.411781675,5.411781675,4.19459421,106.3900763
HoCo3B2,4.966661057,4.966661057,2.9688555,63.42328583
HfCo3B2,4.765324955,4.765324955,3.00897377,59.17442901
HoB2Rh3,5.357541012,5.357541012,3.144437152,78.163601
NpAl3Pd2,5.473691859,5.473691859,4.186524,108.6287813
NdSi2Ru3,5.652501858,5.652501858,3.619016,100.1388414
TbAl3Cu2,5.3132932,5.3132932,4.02791989,98.47795469
CaB2Rh3,5.569696944,5.569696944,2.9138276,78.28122526
LuB2Ir3,5.404053618,5.404053618,3.134074794,79.26460266
UCo3B2,4.959258077,4.959258077,2.99949579,63.88697234
SmMn2Ni3,5.053933899,5.053933899,3.996422,88.40179585
LaNi3Rh2,5.077330102,5.077330102,4.204556,93.86886632
NdB2Ru3,5.570778317,5.570778317,3.048895,81.94167795
LaGa3Pd2,5.443680853,5.443680853,4.2902142,110.1019268
    """, "volume"))
    except Exception as e:
        print(f"[ERROR] - inference : {e}")