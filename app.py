from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "safe_flow_model.joblib"
EVALUATION_PATH = ROOT_DIR / "models" / "evaluation.json"
try:
    MODEL_BUNDLE = joblib.load(MODEL_PATH)
    EVALUATION = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
except (FileNotFoundError, OSError, ValueError, EOFError):
    MODEL_BUNDLE = None
    EVALUATION = None

FEATURES = [
    "network_packet_size", "login_attempts", "session_duration",
    "ip_reputation_score", "failed_logins", "unusual_time_access",
    "protocol_type", "encryption_used", "browser_type",
]
NUMERIC_FEATURES = {
    "network_packet_size", "login_attempts", "session_duration",
    "ip_reputation_score", "failed_logins", "unusual_time_access",
}
FEATURE_LABELS = {
    "network_packet_size": "Packet size",
    "login_attempts": "Login attempts",
    "session_duration": "Session duration",
    "ip_reputation_score": "IP reputation",
    "failed_logins": "Failed logins",
    "unusual_time_access": "Unusual access time",
    "protocol_type": "Protocol",
    "encryption_used": "Encryption",
    "browser_type": "Browser type",
}

FIELDS = {
    "login_attempts": ("Login attempts", 1, 13),
    "failed_logins": ("Failed logins", 0, 5),
    "session_duration": ("Session duration (seconds)", 0, 7200),
    "network_packet_size": ("Network packet size (bytes)", 64, 1500),
    "ip_reputation_score": ("IP reputation score", 0, 1),
    "protocol_type": ("Protocol", None, None),
    "encryption_used": ("Encryption", None, None),
    "browser_type": ("Browser", None, None),
    "unusual_time_access": ("Unusual time access", 0, 1),
}
ALIASES = {
    "login_attempts": ["login attempts", "login_attempts", "attempts", "auth attempts", "authentication attempts"],
    "failed_logins": ["failed logins", "failed_logins", "failed login attempts", "failed attempts", "login failures"],
    "session_duration": ["session duration", "session_duration", "duration", "flow duration"],
    "network_packet_size": ["network packet size", "network_packet_size", "packet size", "packet_length", "packet length"],
    "ip_reputation_score": ["ip reputation score", "ip_reputation_score", "reputation", "ip reputation", "reputation score"],
    "protocol_type": ["protocol", "protocol_type", "protocol type"],
    "encryption_used": ["encryption", "encryption_used", "encryption used"],
    "browser_type": ["browser", "browser_type", "browser type"],
    "unusual_time_access": ["unusual time access", "unusual_time_access", "unusual access", "access time unusual"],
}


def parse_csv_upload(file):
    raw = file.stream.read()
    text = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("This file encoding is not supported. Save it as CSV and try again.")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    try:
        headers = next(reader)
    except StopIteration:
        raise ValueError("This CSV is empty.")
    headers = [str(h).strip() or f"Column {i + 1}" for i, h in enumerate(headers)]
    if not headers:
        raise ValueError("The CSV needs a header row.")
    # DictReader otherwise silently overwrites duplicate/blank header names.
    seen, unique_headers = {}, []
    for header in headers:
        seen[header] = seen.get(header, 0) + 1
        unique_headers.append(header if seen[header] == 1 else f"{header} ({seen[header]})")
    rows = []
    for values in reader:
        if not values or all(not value.strip() for value in values):
            continue
        row = {header: (values[i].strip() if i < len(values) else "") for i, header in enumerate(unique_headers)}
        if len(values) > len(unique_headers):
            row["Extra values"] = ", ".join(values[len(unique_headers):])
        rows.append(row)
        if len(rows) > 50000:
            raise ValueError("This upload has more than 50,000 data rows. Split it into smaller CSV files and retry.")
    if not rows:
        raise ValueError("The CSV has a header but no data rows.")
    return unique_headers, rows


def detect_mapping(headers):
    lookup = {header.strip().lower(): header for header in headers}
    mapping = {}
    for field, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in lookup:
                mapping[field] = lookup[alias.lower()]
                break
    return mapping


def normalize_features(row: dict) -> dict:
    """Coerce incoming form/CSV values to the types used by the trained pipeline."""
    normalized = {}
    for feature in FEATURES:
        value = row.get(feature)
        if value is None or str(value).strip() == "":
            normalized[feature] = None
        elif feature in NUMERIC_FEATURES:
            if feature == "unusual_time_access":
                normalized[feature] = 1.0 if str(value).strip().lower() in {"yes", "true", "1", "unusual"} else 0.0
            else:
                try:
                    number = float(value)
                    normalized[feature] = number if math.isfinite(number) else None
                except (TypeError, ValueError):
                    normalized[feature] = None
        else:
            normalized[feature] = str(value).strip()
    return normalized


def summarize_sensitivity(impacts):
    impacts = [(feature, delta) for feature, delta in impacts if abs(delta) > 0.002]
    impacts.sort(key=lambda item: abs(item[1]), reverse=True)
    strongest = max((abs(delta) for _, delta in impacts), default=0)
    factors = [{
        "label": FEATURE_LABELS[feature],
        "strength": round(abs(delta) / strongest * 100) if strongest else 0,
        "direction": "raises" if delta > 0 else "lowers",
    } for feature, delta in impacts[:4]]
    if factors:
        explanation = (
            f"The score is most sensitive to {', '.join(item['label'].lower() for item in factors[:2])} "
            "when compared with typical training values. This is a sensitivity summary, not a causal explanation."
        )
    else:
        explanation = "No single input produced a clear score shift against typical training values."
    return explanation, factors


def predict(row: dict, explain: bool = True) -> dict:
    """Run the fitted dataset pipeline and optionally calculate local sensitivity."""
    if MODEL_BUNDLE is None:
        raise RuntimeError("The trained SAFE Flow model is missing. Run training/train_model.py first.")
    pipeline = MODEL_BUNDLE["pipeline"]
    normalized = normalize_features(row)
    frame = pd.DataFrame([normalized], columns=FEATURES)
    predicted_class = int(pipeline.predict(frame)[0])
    probabilities = pipeline.predict_proba(frame)[0]
    class_index = list(pipeline.classes_).index(1)
    attack_probability = float(probabilities[class_index])
    confidence = float(max(probabilities))
    factors = []

    if explain:
        baseline = MODEL_BUNDLE["baseline"]
        impacts = []
        for feature in FEATURES:
            if normalized.get(feature) is None:
                continue
            counterfactual = dict(normalized)
            counterfactual[feature] = baseline[feature]
            counterfactual_frame = pd.DataFrame([counterfactual], columns=FEATURES)
            alternate_probability = float(pipeline.predict_proba(counterfactual_frame)[0][class_index])
            delta = attack_probability - alternate_probability
            impacts.append((feature, delta))
        explanation, factors = summarize_sensitivity(impacts)
    else:
        explanation = f"Scored by the trained {EVALUATION['selected_model']} model."

    return {
        "prediction": "Attack" if predicted_class == 1 else "Normal",
        "probability": round(confidence, 6),
        "attack_probability": round(attack_probability, 6),
        "explanation": explanation,
        "factors": factors,
        "feature_importance": [
            {"label": FEATURE_LABELS[item["name"]], "importance": item["importance"]}
            for item in MODEL_BUNDLE.get("feature_importance", [])[:5]
        ],
        "model": EVALUATION["selected_model"],
    }


@app.get("/")
def home(): return render_template("index.html", page="home", evaluation=EVALUATION)

@app.get("/predict")
def predict_page(): return render_template("index.html", page="predict", evaluation=EVALUATION)

@app.get("/about")
def about(): return render_template("index.html", page="about", evaluation=EVALUATION)

@app.get("/relevance")
def relevance(): return render_template("index.html", page="relevance")

@app.post("/api/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    for key, (label, low, high) in FIELDS.items():
        if key in data and low is not None:
            try: value = float(data[key])
            except (TypeError, ValueError): return jsonify(error=f"{label} must be a number."), 400
            if not math.isfinite(value) or value < low or value > high:
                return jsonify(error=f"{label} must be between {low} and {high}."), 400
    try:
        return jsonify(predict(data))
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503


@app.post("/api/inspect-csv")
def inspect_csv_api():
    file = request.files.get("file")
    if not file:
        return jsonify(error="Choose a CSV file to inspect."), 400
    try:
        headers, rows = parse_csv_upload(file)
        return jsonify(filename=file.filename or "uploaded.csv", headers=headers, total=len(rows),
                       preview=rows[:5], auto_mapping=detect_mapping(headers))
    except (ValueError, csv.Error) as exc:
        return jsonify(error=str(exc) or "Could not read this CSV."), 400

@app.post("/api/upload")
def upload_api():
    file = request.files.get("file")
    if not file:
        return jsonify(error="Choose a CSV file to analyze."), 400
    try:
        headers, rows = parse_csv_upload(file)
        try:
            requested_mapping = json.loads(request.form.get("mapping", "{}"))
        except json.JSONDecodeError:
            requested_mapping = {}
        available = set(headers)
        mapping = {field: source for field, source in requested_mapping.items()
                   if field in FIELDS and source in available}
        if not mapping:
            mapping = detect_mapping(headers)
        results = []
        if not mapping:
            results = [{"row": index, "source": row, "prediction": "Needs mapping", "probability": None,
                        "attack_probability": None, "explanation": "No columns were mapped to session features."}
                       for index, row in enumerate(rows, 1)]
        else:
            if MODEL_BUNDLE is None:
                return jsonify(error="The trained SAFE Flow model is missing. Run training/train_model.py first."), 503
            feature_rows = [
                normalize_features({field: row.get(source, "") for field, source in mapping.items()})
                for row in rows
            ]
            frame = pd.DataFrame(feature_rows, columns=FEATURES)
            pipeline = MODEL_BUNDLE["pipeline"]
            predictions = pipeline.predict(frame)
            probabilities = pipeline.predict_proba(frame)
            class_index = list(pipeline.classes_).index(1)
            base_attack_scores = probabilities[:, class_index]
            impacts_by_row = [[] for _ in rows]
            baseline = MODEL_BUNDLE["baseline"]
            # Vectorized, one-feature-at-a-time sensitivity for the batch explanation column.
            for feature in mapping:
                altered = frame.copy()
                altered[feature] = baseline[feature]
                alternate_scores = pipeline.predict_proba(altered)[:, class_index]
                for row_index, (base_score, alternate_score) in enumerate(zip(base_attack_scores, alternate_scores)):
                    impacts_by_row[row_index].append((feature, float(base_score - alternate_score)))
            for index, (row, predicted_class, scores) in enumerate(zip(rows, predictions, probabilities), 1):
                explanation, _ = summarize_sensitivity(impacts_by_row[index - 1])
                results.append({
                    "row": index,
                    "source": row,
                    "prediction": "Attack" if int(predicted_class) == 1 else "Normal",
                    "probability": round(float(max(scores)), 6),
                    "attack_probability": round(float(scores[class_index]), 6),
                    "explanation": explanation,
                })
        return jsonify(results=results, total=len(results), attacks=sum(r["prediction"] == "Attack" for r in results),
                       normals=sum(r["prediction"] == "Normal" for r in results),
                       unmapped=sum(r["prediction"] == "Needs mapping" for r in results),
                       headers=headers, mapping=mapping, mapped_count=len(mapping), filename=file.filename or "uploaded.csv")
    except (ValueError, csv.Error) as exc:
        return jsonify(error=str(exc) or "Could not read this CSV."), 400


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="This file is larger than 64 MB. Save a smaller CSV or split it into files."), 413

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
