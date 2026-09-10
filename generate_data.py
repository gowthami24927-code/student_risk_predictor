import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

attendance  = np.random.randint(30, 100, N)
marks       = np.random.randint(20, 100, N)
assignments = np.random.randint(0, 11, N)
late        = np.random.randint(0, 11, N)
failures    = np.random.randint(0, 5, N)
trend       = np.random.choice(["Increasing", "Stable", "Decreasing"], N)


def att_cat(v):   return "Low" if v < 60 else "Medium" if v < 75 else "High"
def marks_cat(v): return "Low" if v < 50 else "Medium" if v < 75 else "High"
def assign_cat(v):return "Low" if v < 5  else "Medium" if v < 8  else "High"
def late_cat(v):  return "High" if v > 3 else "Low"
def fail_cat(v):  return "High" if v > 1 else "Low"


att_c    = [att_cat(v)    for v in attendance]
marks_c  = [marks_cat(v)  for v in marks]
assign_c = [assign_cat(v) for v in assignments]
late_c   = [late_cat(v)   for v in late]
fail_c   = [fail_cat(v)   for v in failures]


# Absenteeism_Risk depends on Attendance + Trend
def absenteeism_risk(att, tr):
    p_high = {"Low": 0.85, "Medium": 0.40, "High": 0.10}[att]
    if tr == "Decreasing": p_high = min(p_high + 0.10, 0.95)
    if tr == "Increasing": p_high = max(p_high - 0.10, 0.02)
    r = np.random.random()
    if r < p_high:
        return "High"
    elif r < p_high + (1 - p_high) * 0.5:
        return "Medium"
    else:
        return "Low"


# Late_Submission_Risk depends on Assignments + Late
def late_submission_risk(asgn, lt):
    p_high = 0.10
    if lt == "High":   p_high += 0.60
    if asgn == "Low":  p_high += 0.20
    p_high = min(p_high, 0.95)
    r = np.random.random()
    return "High" if r < p_high else "Low"


# Dropout_Risk depends on Absenteeism_Risk + Late_Submission_Risk + Marks + Failures
def dropout_risk(abs_r, late_r, mrk, fail):
    p = 0.05
    if abs_r == "High":   p += 0.35
    elif abs_r == "Medium": p += 0.15
    if late_r == "High":  p += 0.20
    if mrk == "Low":      p += 0.25
    elif mrk == "Medium": p += 0.10
    if fail == "High":    p += 0.20
    p = min(p, 0.97)
    r = np.random.random()
    if r < p * 0.65:
        return "High"
    elif r < p * 0.65 + (1 - p) * 0.55:
        return "Medium"
    else:
        return "Low"


abs_risk  = [absenteeism_risk(att_c[i], trend[i]) for i in range(N)]
late_risk = [late_submission_risk(assign_c[i], late_c[i]) for i in range(N)]
drop_risk = [dropout_risk(abs_risk[i], late_risk[i], marks_c[i], fail_c[i]) for i in range(N)]

df = pd.DataFrame({
    "Student_ID":           range(1, N + 1),
    "Attendance_Percent":   attendance,
    "Average_Marks":        marks,
    "Assignments_Submitted":assignments,
    "Late_Submissions":     late,
    "Previous_Failures":    failures,
    "Attendance_Trend":     trend,
    "Absenteeism_Risk":     abs_risk,
    "Late_Submission_Risk": late_risk,
    "Dropout_Risk":         drop_risk,
})

df.to_csv("students.csv", index=False)
print(f"Generated {N} rows")
print(df["Dropout_Risk"].value_counts())
print(df["Absenteeism_Risk"].value_counts())
print(df["Late_Submission_Risk"].value_counts())
