

from pyexpat import model
import re
import ast
from api_key import *
import json as pyjson
from llm_inference import *
from api_key import OPENAI_API_KEY
import os
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage

RUBRIC = [
	{
		"name": "Mechanistic alignment",
		"weight": 0.25,
		"description": "Does the property (or relationship) arise from the same underlying physics/chemistry on both sides of the analogy? 5 = mechanisms are identical / same causal drivers. 3 = mechanisms partially similar. 0 = mechanisms different or opposed."
	},
	{
		"name": "Mapping fidelity",
		"weight": 0.25,
		"description": "Does the way A → B changes the property map consistently to how C → D would be expected to change? 5 = difference maps cleanly and causally. 3 = partial mapping. 0 = no meaningful correspondence."
	},
	{
		"name": "Family",
		"weight": 0.20,
		"description": "Are the materials in the same family/design space (bonding type, crystal structure, chemical class) so trends are transferable? 5 = same family/systematic trend. 2 = weak resemblance. 0 = different classes."
	},
	{
		"name": "Competing mechanisms",
		"weight": 0.15,
		"description": "Is the property governed by a single dominant mechanism, or are multiple confounders at play? 5 = property dominated by one mechanism on both sides. 0 = multiple competing mechanisms swamp the analogy."
	}
]

class Grading():
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

	def llm_grade_analogy(self, analogy, model="gpt-5-mini"):
		prompt = f"""
You are an expert in materials science and analogy evaluation. 
Given the analogy and rubric below, assign a score (0-5) 
for each category and provide a brief justification for each. Respond in JSON as:
{{
	"Mechanistic alignment": {{"score": int, "justification": str}},
	"Mapping fidelity": {{"score": int, "justification": str}},
	"Family": {{"score": int, "justification": str}},
	"Competing mechanisms": {{"score": int, "justification": str}}
}}

Analogy:
"""
		prompt += analogy + "\n\nRubric:\n"
		for rubric in RUBRIC:
			prompt += f"- {rubric['name']}: {rubric['description']}\n"
		os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
		llm = init_chat_model(model, model_provider="openai")
		messages = [
			HumanMessage(content=prompt),
		]
		out = llm.invoke(messages)
		text = out.content
		match = re.search(r'\{[\s\S]*\}', text)
		if match:
			json_str = match.group(0)
			scores = pyjson.loads(json_str)
			return scores
		else:
			raise ValueError("not found " + text)

	def compute_score(self, scores):
		total = 0
		for i, v in enumerate(RUBRIC):
			total += scores[v["name"]]["score"] * v["weight"]
		percent = (total / 5) * 100
		letter_grade = self.get_letter_grade(percent)
		return total, percent, letter_grade

if __name__ == "__main__":
    analogy = """Construct an analogy "
    "of the form A is to B as C is to D in order to find the property of D """
    model = "gpt-5-mini"
    grading = Grading()
    scores = grading.llm_grade_analogy(analogy, model='gpt-5-mini')
    total, percent, letter_grade = grading.compute_score(scores)
    manifest = {
        "analogy": analogy,
        "scores": scores,
        "weighted_score": total,
        "percent": percent,
        "letter_grade": letter_grade
}
    with open("grading_manifest.json", "w") as f:
        pyjson.dump(manifest, f, indent=2)
        print("make grading_manifest.json")
