from flask import Flask, render_template, request, jsonify
import pickle
from pgmpy.inference import VariableElimination

app = Flask(__name__)

with open("bayesian_dropout_model.pkl", "rb") as f:
    model = pickle.load(f)

inference = VariableElimination(model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    attendance = float(data["attendance"])
    marks      = float(data["marks"])
    assignments = int(data["assignments"])
    late       = int(data["late_submissions"])
    failures   = int(data["failures"])
    trend      = data["trend"]

    att_cat   = "Low" if attendance < 60 else "Medium" if attendance < 75 else "High"
    marks_cat = "Low" if marks < 50 else "Medium" if marks < 75 else "High"
    assign_cat = "Low" if assignments < 5 else "Medium" if assignments < 8 else "High"
    late_cat  = "High" if late > 3 else "Low"
    fail_cat  = "High" if failures > 1 else "Low"

    evidence = {
        "Attendance":  att_cat,
        "Marks":       marks_cat,
        "Assignments": assign_cat,
        "Late":        late_cat,
        "Failures":    fail_cat,
        "Trend":       trend
    }

    def get_prob(result, variable, state):
        states = result.state_names[variable]
        for i, s in enumerate(states):
            if s == state:
                return result.values[i]
        return 0.0

    absenteeism_result = inference.query(["Absenteeism_Risk"], evidence=evidence)
    late_result        = inference.query(["Late_Submission_Risk"], evidence=evidence)
    dropout_result     = inference.query(["Dropout_Risk"], evidence=evidence)

    absenteeism_probability = round(get_prob(absenteeism_result, "Absenteeism_Risk", "High") * 100, 2)
    late_probability        = round(get_prob(late_result, "Late_Submission_Risk", "High") * 100, 2)
    dropout_probability     = round(get_prob(dropout_result, "Dropout_Risk", "High") * 100, 2)

    if dropout_probability >= 70:
        risk_level = "HIGH"
    elif dropout_probability >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    reasons = []
    if attendance < 75:
        reasons.append("Low attendance")
    if marks < 50:
        reasons.append("Low average marks")
    if assignments < 5:
        reasons.append("Few assignments submitted")
    if late > 3:
        reasons.append("High number of late submissions")
    if failures > 1:
        reasons.append("Previous academic failures")
    if trend == "Decreasing":
        reasons.append("Attendance trend is decreasing")
    if not reasons:
        reasons.append("No major risk factors found")

    if risk_level == "HIGH":
        recommendation = "Immediate academic intervention is recommended. Faculty should monitor the student regularly."
    elif risk_level == "MEDIUM":
        recommendation = "The student should be monitored regularly. Encourage better attendance and timely submission."
    else:
        recommendation = "Student is currently at low risk. Continue normal academic monitoring."

    return jsonify({
        "absenteeism_probability": absenteeism_probability,
        "late_probability":        late_probability,
        "dropout_probability":     dropout_probability,
        "risk_level":              risk_level,
        "reasons":                 reasons,
        "recommendation":          recommendation
    })


if __name__ == "__main__":
    app.run(debug=True)
