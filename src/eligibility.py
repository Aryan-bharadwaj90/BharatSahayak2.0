import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMES_FILE = ROOT / "data" / "schemes.json"


def _load_schemes():
    with open(SCHEMES_FILE, encoding="utf-8") as f:
        return json.load(f)


def _compare(actual, operator, expected):
    if operator == "eq": return actual == expected
    if operator == "gte": return actual >= expected
    if operator == "lte": return actual <= expected
    if operator == "gt": return actual > expected
    if operator == "lt": return actual < expected
    if operator == "in": return actual in expected
    if operator == "truthy":
        if not isinstance(expected, bool):
            raise ValueError("truthy requires a boolean expected value")
        return bool(actual) == expected
    raise ValueError(f"Unsupported eligibility operator: {operator}")


def check_eligibility(scheme_id, user_profile):
    scheme = next((s for s in _load_schemes() if s["scheme_id"] == scheme_id), None)
    disclaimer = (
        "Preliminary guidance only. This is not an official eligibility decision; "
        "the relevant government authority/database makes the final determination."
    )
    if scheme is None:
        return {"scheme_id": scheme_id, "status": "unknown", "satisfied": [],
                "not_satisfied": [], "unknown": ["Scheme rules are not configured."],
                "disclaimer": disclaimer}

    rules = scheme.get("eligibility_rules")
    if not isinstance(rules, dict) or not isinstance(rules.get("conditions"), list) or not rules["conditions"]:
        return {"scheme_id": scheme_id, "status": "unknown", "satisfied": [],
                "not_satisfied": [],
                "unknown": ["Verified decision rules are not configured for this scheme."],
                "disclaimer": disclaimer}

    profile = user_profile or {}
    if scheme_id == "PMAY_URBAN":
        legacy = "previous_pmAY_urban_house_curtailed_after_2023_12_31"
        if legacy in profile and legacy.lower() not in profile:
            profile = {**profile, legacy.lower(): profile[legacy]}

    satisfied, not_satisfied, unknown = [], [], []
    for index, condition in enumerate(rules["conditions"], start=1):
        if not isinstance(condition, dict):
            unknown.append(f"Invalid eligibility rule {index}.")
            continue
        field = condition.get("field")
        if (not isinstance(field, str) or not field.strip()
                or not isinstance(condition.get("operator"), str) or condition.get("value") is None):
            unknown.append(f"Invalid eligibility rule {index}. Check field, operator and value.")
            continue
        reason = condition.get("reason", field)
        actual = profile.get(field)
        if actual is None or actual == "":
            unknown.append(f"Please provide/verify: {field}. {reason}")
            continue
        try:
            passed = _compare(actual, condition["operator"], condition["value"])
        except (TypeError, ValueError):
            unknown.append(f"Could not safely evaluate: {field}. {reason}")
            continue
        (satisfied if passed else not_satisfied).append(reason)

    logic = rules.get("logic", "all")
    if logic == "all":
        status = "not_satisfied" if not_satisfied else ("unknown" if unknown else "satisfied")
    elif logic == "any":
        status = "satisfied" if satisfied else ("unknown" if unknown else "not_satisfied")
    else:
        unknown.append(f"Unsupported eligibility rule logic: {logic}")
        status = "unknown"

    notes = rules.get("notes")
    if notes:
        unknown.append(f"Important limitation: {notes}")
    return {"scheme_id": scheme_id, "scheme_name": scheme.get("scheme_name"),
            "status": status, "satisfied": satisfied,
            "not_satisfied": not_satisfied, "unknown": unknown,
            "disclaimer": disclaimer}
