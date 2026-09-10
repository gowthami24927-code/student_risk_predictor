import pandas as pd
import pickle
from pgmpy.models import DiscreteBayesianNetwork

data = pd.read_csv("students.csv")


def att_cat(v):
    return "Low" if v < 60 else "Medium" if v < 75 else "High"

def marks_cat(v):
    return "Low" if v < 50 else "Medium" if v < 75 else "High"

def assign_cat(v):
    return "Low" if v < 5 else "Medium" if v < 8 else "High"

def late_cat(v):
    return "High" if v > 3 else "Low"

def fail_cat(v):
    return "High" if v > 1 else "Low"


data["Attendance"]  = data["Attendance_Percent"].apply(att_cat)
data["Marks"]       = data["Average_Marks"].apply(marks_cat)
data["Assignments"] = data["Assignments_Submitted"].apply(assign_cat)
data["Late"]        = data["Late_Submissions"].apply(late_cat)
data["Failures"]    = data["Previous_Failures"].apply(fail_cat)
data["Trend"]       = data["Attendance_Trend"]

data = data[[
    "Attendance", "Marks", "Assignments", "Late", "Failures", "Trend",
    "Absenteeism_Risk", "Late_Submission_Risk", "Dropout_Risk"
]]

# Intermediate nodes reduce parent combinations per node
# Dropout_Risk now has only 4 parents instead of 6
model = DiscreteBayesianNetwork([
    ("Attendance",           "Absenteeism_Risk"),
    ("Trend",                "Absenteeism_Risk"),
    ("Assignments",          "Late_Submission_Risk"),
    ("Late",                 "Late_Submission_Risk"),
    ("Absenteeism_Risk",     "Dropout_Risk"),
    ("Late_Submission_Risk", "Dropout_Risk"),
    ("Marks",                "Dropout_Risk"),
    ("Failures",             "Dropout_Risk"),
])

model.fit(data)

with open("bayesian_dropout_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Bayesian Network trained and saved.")
