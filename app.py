"""
app.py
Production-style Flask API for insurance claim outcome prediction.
Defensively validates inputs and prevents server crashes on malformed requests.
"""

import math
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
# Preserve dict insertion order in json responses (disable alphabetical key sorting)
app.json.sort_keys = False

# Load model pipeline once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
else:
    model_pipeline = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_pipeline is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Model is not loaded. Run train.py first."}), 500

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # 1. Check required fields
    required_fields = ["payer", "provider", "cpt_code", "diagnosis_code", "billed_amount", "patient_age"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    billed_amount = data["billed_amount"]
    patient_age = data["patient_age"]
    payer = data["payer"]
    provider = data["provider"]
    cpt_code = data["cpt_code"]
    diagnosis_code = data["diagnosis_code"]

    # 2. Validate billed_amount: numeric, finite, positive, reject booleans
    if isinstance(billed_amount, bool) or not isinstance(billed_amount, (int, float)):
        return jsonify({"error": "'billed_amount' must be a numeric value"}), 400
    if not math.isfinite(billed_amount):
        return jsonify({"error": "'billed_amount' must be finite (NaN/Infinity not allowed)"}), 400
    if billed_amount <= 0:
        return jsonify({"error": "'billed_amount' must be greater than 0"}), 400

    # 3. Validate patient_age: strictly integer, reject bool and non-int floats, sane range 0-120
    if isinstance(patient_age, bool) or not isinstance(patient_age, int):
        return jsonify({"error": "'patient_age' must be an integer"}), 400
    if not (0 <= patient_age <= 120):
        return jsonify({"error": "'patient_age' must be between 0 and 120"}), 400

    # 4. Validate string fields (non-empty after trimming whitespace)
    for name, val in [("payer", payer), ("provider", provider), ("diagnosis_code", diagnosis_code)]:
        if not isinstance(val, str) or not val.strip():
            return jsonify({"error": f"'{name}' must be a non-empty string"}), 400

    # CPT code can be provided as string ("99213") or integer (99213)
    if isinstance(cpt_code, bool):
        return jsonify({"error": "'cpt_code' cannot be a boolean"}), 400
    if isinstance(cpt_code, (int, str)):
        cpt_code_str = str(cpt_code).strip()
        if not cpt_code_str:
            return jsonify({"error": "'cpt_code' cannot be empty"}), 400
    else:
        return jsonify({"error": "'cpt_code' must be a string or integer"}), 400

    # 5. Build 1-row DataFrame for inference (strip text inputs; unknown categories handled by pipeline)
    claim_df = pd.DataFrame([{
        "payer": payer.strip(),
        "provider": provider.strip(),
        "cpt_code": cpt_code_str,
        "diagnosis_code": diagnosis_code.strip(),
        "billed_amount": float(billed_amount),
        "patient_age": int(patient_age),
    }])

    try:
        pred_label = model_pipeline.predict(claim_df)[0]
        raw_probs = model_pipeline.predict_proba(claim_df)[0]

        # Map probabilities using model.classes_ dynamic indices and present in spec order
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(model_pipeline.classes_)}
        ordered_classes = ["Paid", "Partially_Paid", "Denied"]

        probabilities = {}
        for cls_name in ordered_classes:
            if cls_name in class_to_idx:
                probabilities[cls_name] = round(float(raw_probs[class_to_idx[cls_name]]), 2)

        return jsonify({
            "predicted_status": pred_label,
            "probabilities": probabilities,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
