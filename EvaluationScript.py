import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

INPUT_FOLDER = "Output/Chair_timeseries"
SUMMARY_OUTPUT = "Output/optic_flow_summary.csv"

all_trials = []
summary_rows = []

# -----------------------------
# Load all CSV files
# -----------------------------
for file in os.listdir(INPUT_FOLDER):

    if not file.endswith(".csv"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)

    # Ignore metadata header lines beginning with "#"
    df = pd.read_csv(filepath, comment="#")

    video_name = file.replace("_flow_timeseries.csv", "")

    # Determine condition from filename
    if "Center" in video_name:
        condition = "Center"
    elif "Whole" in video_name:
        condition = "Whole"
    else:
        condition = "Unknown"

    df["video"] = video_name
    df["condition"] = condition

    all_trials.append(df)

    # -----------------------------
    # Summary statistics per video
    # -----------------------------
    summary_rows.append({
        "video": video_name,
        "condition": condition,
        "mean_flow": df["mean_mag"].mean(),
        "median_flow": df["mean_mag"].median(),
        "max_flow": df["mean_mag"].max(),
        "sd_flow": df["mean_mag"].std(),
        "mean_p90": df["p90_mag"].mean(),
        "median_p90": df["p90_mag"].median(),
        "mean_pxsec": df["mean_mag_px_per_sec"].mean(),
        "duration_sec": df["time_sec"].max()
    })

# Combine
all_data = pd.concat(all_trials, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

# Save summary table
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

print("\nSummary table:")
print(summary_df)

# -----------------------------
# Plot: Mean optic flow boxplot
# -----------------------------
plt.figure(figsize=(6,5))
sns.boxplot(data=summary_df, x="condition", y="mean_flow")
sns.stripplot(data=summary_df, x="condition", y="mean_flow", color="black")
plt.title("Mean Optic Flow by Condition")
plt.ylabel("Mean Flow Magnitude")
plt.xlabel("")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot : p90 optic flow boxplot
# -----------------------------
plt.figure(figsize=(6,5))
sns.boxplot(data=summary_df, x="condition", y="mean_p90")
sns.stripplot(data=summary_df, x="condition", y="mean_p90", color="black")
plt.title("High Motion (p90) by Condition")
plt.ylabel("Mean p90 Flow Magnitude")
plt.xlabel("")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: mean px sec optic flow boxplot
# -----------------------------
plt.figure(figsize=(6,5))
sns.boxplot(data=summary_df, x="condition", y="mean_pxsec")
sns.stripplot(data=summary_df, x="condition", y="mean_pxsec", color="black")
plt.title("Mean optic flow (pixels/sec) by Condition")
plt.ylabel("Mean Flow Magnitude (pixels/sec)")
plt.xlabel("")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Time series overlay
# -----------------------------
# plt.figure(figsize=(8,5))
#
# for video, df in all_data.groupby("video"):
#     cond = df["condition"].iloc[0]
#     if cond == "Center":
#         color = "blue"
#     else:
#         color = "red"
#
#     plt.plot(df["time_sec"], df["mean_mag"], color=color, alpha=0.4)
#
# plt.xlabel("Time (sec)")
# plt.ylabel("Mean Flow Magnitude")
# plt.title("Optic Flow Time Series per Trial")
# plt.tight_layout()
# plt.show()

# -----------------------------
# Plot: Average time series by condition
# -----------------------------
# plt.figure(figsize=(8,5))
#
# for cond, df in all_data.groupby("condition"):
#
#     # Bin time to align trials
#     df["time_bin"] = df["time_sec"].round(1)
#
#     avg = df.groupby("time_bin")["mean_mag"].mean()
#
#     plt.plot(avg.index, avg.values, label=cond)
#
# plt.xlabel("Time (sec)")
# plt.ylabel("Mean Flow Magnitude")
# plt.title("Average Optic Flow Over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()

# -----------------------------
# Statistical Test: t-test
# -----------------------------
# Load the summary table
summary_df = pd.read_csv("Output/optic_flow_summary.csv")

# Split conditions
center = summary_df[summary_df["condition"] == "Center"]["mean_flow"]
whole = summary_df[summary_df["condition"] == "Whole"]["mean_flow"]

print("\nCenter values:")
print(center.values)

print("\nWhole values:")
print(whole.values)

# Independent samples t-test
t_stat, p_val = ttest_ind(center, whole, equal_var=False)

print("\nT-test results")
print("t =", t_stat)
print("p =", p_val)