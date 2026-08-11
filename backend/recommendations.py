import json
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOMMENDATIONS_PATH = PROJECT_ROOT / "knowledge_base" / "recommendations.json"


@lru_cache(maxsize=1)
def load_recommendations():
    if not RECOMMENDATIONS_PATH.exists():
        return {}
    return json.loads(RECOMMENDATIONS_PATH.read_text(encoding="utf-8"))


def get_recommendation(class_name: str):
    recommendations = load_recommendations()
    default = recommendations.get(
        "default",
        {
            "display_name": class_name.replace("___", " - ").replace("_", " "),
            "organic_treatment": "Use local agricultural guidance before treatment.",
            "chemical_treatment": "Consult an agricultural officer before applying chemicals.",
            "prevention": "Upload a clearer leaf image if the prediction confidence is low.",
        },
    )
    return recommendations.get(class_name, default)
