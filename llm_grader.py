from api_key import ANTHROPIC_API_KEY, GOOGLE_GENAI_API_KEY, HUGGINGFACE_API_KEY, OPENAI_API_KEY
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, Union

MechanisticAlignmentDescr = Annotated[str, Field(description="Does the property (or relationship) arise from the same underlying physics/chemistry on both sides of the analogy? 5 = mechanisms are identical / same causal drivers. 3 = mechanisms partially similar. 0 = mechanisms different or opposed.")]
MechanisticAlignmentScore = Annotated[int, Field(description="Mechanistic alignment integer score, 0-5.")]
MappingFidelityDescr = Annotated[str, Field(description="Does the way A → B changes the property map consistently to how C → D would be expected to change? 5 = difference maps cleanly and causally. 3 = partial mapping. 0 = no meaningful correspondence.")]
MappingFidelityScore = Annotated[int, Field(description="Mapping fidelity integer score, 0-5.")]
FamilyDescr = Annotated[str, Field(description="Are the materials in the same family/design space (bonding type, crystal structure, chemical class) so trends are transferable? 5 = same family/systematic trend. 2 = weak resemblance. 0 = different classes.")]
FamilyScore = Annotated[int, Field(description="Family score integer score, 0-5.")]
CompetingMechanismsDescr = Annotated[str, Field(description="Is the property governed by a single dominant mechanism, or are multiple confounders at play? 5 = property dominated by one mechanism on both sides. 0 = multiple competing mechanisms swamp the analogy.")]
CompetingMechanismsScore = Annotated[int, Field(description="Competing mechanisms integer score, 0-5.")]

RUBRIC = [
	{
		"name": "Mechanistic Alignment",
		"weight": 0.25,
		"description": "Does the property (or relationship) arise from the same underlying physics/chemistry on both sides of the analogy? 5 = mechanisms are identical / same causal drivers. 3 = mechanisms partially similar. 0 = mechanisms different or opposed."
	},
	{
		"name": "Mapping Fidelity",
		"weight": 0.25,
		"description": "Does the way A → B changes the property map consistently to how C → D would be expected to change? 5 = difference maps cleanly and causally. 3 = partial mapping. 0 = no meaningful correspondence."
	},
	{
		"name": "Family",
		"weight": 0.25,
		"description": "Are the materials in the same family/design space (bonding type, crystal structure, chemical class) so trends are transferable? 5 = same family/systematic trend. 2 = weak resemblance. 0 = different classes."
	},
	{
		"name": "Competing Mechanisms",
		"weight": 0.25,
		"description": "Is the property governed by a single dominant mechanism, or are multiple confounders at play? 5 = property dominated by one mechanism on both sides. 0 = multiple competing mechanisms swamp the analogy."
	}
]

SYSTEM_PROMPT = (
"You are an expert in materials science and analogy evaluation. "
"Given the analogy and rubric below, assign a score (0-5) "
"for each category and provide a brief justification for each."
)

class GradeResponse(BaseModel):
    mechanistic_alignment_descr: MechanisticAlignmentDescr
	mechanistic_alignment_score: MechanisticAlignmentScore
    mapping_fidelity_descr: MappingFidelityDescr
	mapping_fidelity_score: MappingFidelityScore
    family_descr: FamilyDescr
	family_score: FamilyScore
    competing_mechanisms_descr: CompetingMechanismsDescr
	competing_mechanisms_score: CompetingMechanismsScore

def compute_score(self, scores):
	total = 0
	for i, v in enumerate(RUBRIC):
		total += scores[v["name"]]["score"] * v["weight"]
	percent = (total / 5) * 100
	letter_grade = self.get_letter_grade(percent)
	return total, percent, letter_grade

def get_letter_grade(self, percent):
	if percent >= 90:
		return "A"
	elif percent >= 75:
		return "B"
	elif percent >= 60:
		return "C"
	elif percent >= 40:
		return "D"
	else:
		return "F"

def call_openai(prompt: str, model: str = "gpt-5-mini")
    llm = init_chat_model(model, model_provider="openai", openai_api_key=OPENAI_API_KEY)
    messages = [
        SystemMessage(content = SYSTEM_PROMPT),
        HumanMessage(content = prompt)
    ]
    out = llm.with_structured_output(GradeResponse).invoke(messages)
    return out