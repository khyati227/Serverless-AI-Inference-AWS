from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ZIP_PATH = Path("results.zip")
EXTRACT_FOLDER = Path("extracted_results")
MEMORY_CONFIGURATIONS = [512, 1024, 2048, 3008]


# ---------------------------------------------------------
# Extract the ZIP file
# ---------------------------------------------------------

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {ZIP_PATH}. "
        "Place results.zip in the same folder as this script."
    )

EXTRACT_FOLDER.mkdir(exist_ok=True)

with ZipFile(ZIP_PATH, "r") as zip_file:
    zip_file.extractall(EXTRACT_FOLDER)


# ---------------------------------------------------------
# Read every CSV and label it
# ---------------------------------------------------------

dataframes = []

for csv_path in EXTRACT_FOLDER.rglob("*.csv"):
    filename = csv_path.name.lower()

    memory_mb = next(
        (
            memory
            for memory in MEMORY_CONFIGURATIONS
            if str(memory) in filename
        ),
        None,
    )

    if memory_mb is None:
        print(f"Skipping file with unknown memory size: {csv_path.name}")
        continue

    start_type = "warm" if "warm" in filename else "cold"

    dataframe = pd.read_csv(csv_path)

    dataframe["memory_mb"] = memory_mb
    dataframe["start_type"] = start_type
    dataframe["source_file"] = csv_path.name

    dataframes.append(dataframe)


if not dataframes:
    raise RuntimeError("No CSV files were found in results.zip.")

master_data = pd.concat(dataframes, ignore_index=True)


# Make sure the measurement columns are numeric.
numeric_columns = [
    "latency_ms",
    "billed_ms",
    "memory_size_mb",
    "memory_used_mb",
    "cold_start_ms",
]

for column in numeric_columns:
    if column in master_data.columns:
        master_data[column] = pd.to_numeric(
            master_data[column],
            errors="coerce",
        )


print("\nSamples found:")
print(
    master_data
    .groupby(["memory_mb", "start_type"])
    .size()
    .rename("samples")
)

print("\nCombined rows:", len(master_data))

# ---------------------------------------------------------
# Descriptive statistics and 95% confidence intervals
# ---------------------------------------------------------

def calculate_statistics(values: pd.Series) -> pd.Series:
    """
    Calculate descriptive statistics and a two-sided
    95% confidence interval for the population mean.

    The interval uses the Student t distribution because
    each experimental group has a small sample size.
    """

    clean_values = values.dropna().astype(float)
    sample_size = len(clean_values)

    if sample_size == 0:
        return pd.Series({
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std_dev": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        })

    mean = clean_values.mean()
    median = clean_values.median()
    standard_deviation = clean_values.std(ddof=1)
    minimum = clean_values.min()
    maximum = clean_values.max()

    if sample_size > 1:
        standard_error = stats.sem(clean_values)
        margin_of_error = (
            stats.t.ppf(0.975, df=sample_size - 1)
            * standard_error
        )

        ci95_low = mean - margin_of_error
        ci95_high = mean + margin_of_error
    else:
        ci95_low = np.nan
        ci95_high = np.nan

    return pd.Series({
        "n": sample_size,
        "mean": mean,
        "median": median,
        "std_dev": standard_deviation,
        "minimum": minimum,
        "maximum": maximum,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    })


latency_statistics = (
    master_data
    .groupby(["memory_mb", "start_type"])["latency_ms"]
    .apply(calculate_statistics)
    .unstack()
    .reset_index()
)

print("\nLatency statistics:")
print(latency_statistics.round(2).to_string(index=False))


cold_start_statistics = (
    master_data[master_data["start_type"] == "cold"]
    .groupby("memory_mb")["cold_start_ms"]
    .apply(calculate_statistics)
    .unstack()
    .reset_index()
)

print("\nCold-start initialization statistics:")
print(cold_start_statistics.round(2).to_string(index=False))


billed_duration_statistics = (
    master_data
    .groupby(["memory_mb", "start_type"])["billed_ms"]
    .apply(calculate_statistics)
    .unstack()
    .reset_index()
)

print("\nBilled-duration statistics:")
print(billed_duration_statistics.round(2).to_string(index=False))
#
# # ---------------------------------------------------------
# # Figure 1: Cold and warm execution latency
# # ---------------------------------------------------------
#
# plot_data = (
#     master_data
#     .groupby(["memory_mb", "start_type"])["latency_ms"]
#     .agg(
#         mean="mean",
#         standard_error="sem",
#         sample_size="count",
#     )
#     .reset_index()
# )
#
# # Student-t 95% confidence interval margin.
# plot_data["ci95_margin"] = plot_data.apply(
#     lambda row: (
#         stats.t.ppf(
#             0.975,
#             df=int(row["sample_size"]) - 1,
#         )
#         * row["standard_error"]
#     ),
#     axis=1,
# )
#
#
# fig, axis = plt.subplots(figsize=(8.5, 5.5))
#
# for start_type in ["cold", "warm"]:
#     subset = (
#         plot_data[plot_data["start_type"] == start_type]
#         .sort_values("memory_mb")
#     )
#
#     axis.errorbar(
#         subset["memory_mb"],
#         subset["mean"],
#         yerr=subset["ci95_margin"],
#         marker="o",
#         linewidth=2,
#         markersize=7,
#         capsize=5,
#         label=f"{start_type.title()} start",
#     )
#
#
# axis.set_title(
#     "AWS Lambda Inference Latency by Memory Allocation",
#     fontsize=15,
#     fontweight="bold",
#     pad=14,
# )
#
# axis.set_xlabel(
#     "Configured Lambda memory (MB)",
#     fontsize=11,
# )
#
# axis.set_ylabel(
#     "Mean handler latency (ms)",
#     fontsize=11,
# )
#
# axis.set_xticks(MEMORY_CONFIGURATIONS)
#
# axis.grid(
#     axis="y",
#     linestyle="--",
#     alpha=0.35,
# )
#
# axis.legend(
#     title="Execution type",
#     frameon=False,
# )
#
# axis.spines["top"].set_visible(False)
# axis.spines["right"].set_visible(False)
#
# fig.text(
#     0.5,
#     0.01,
#     "Points show sample means; error bars show 95% t-confidence intervals.",
#     ha="center",
#     fontsize=9,
# )
#
# fig.tight_layout(rect=[0, 0.04, 1, 1])
#
# plt.show()
# ---------------------------------------------------------
# Figure 2: Cold-start initialization duration
# ---------------------------------------------------------
#
# cold_initialization_plot = (
#     master_data[master_data["start_type"] == "cold"]
#     .groupby("memory_mb")["cold_start_ms"]
#     .agg(
#         mean="mean",
#         standard_error="sem",
#         sample_size="count",
#     )
#     .reset_index()
#     .sort_values("memory_mb")
# )
#
# cold_initialization_plot["ci95_margin"] = (
#     cold_initialization_plot.apply(
#         lambda row: (
#             stats.t.ppf(
#                 0.975,
#                 df=int(row["sample_size"]) - 1,
#             )
#             * row["standard_error"]
#         ),
#         axis=1,
#     )
# )
#
#
# fig, axis = plt.subplots(figsize=(8.5, 5.5))
#
# axis.errorbar(
#     cold_initialization_plot["memory_mb"],
#     cold_initialization_plot["mean"],
#     yerr=cold_initialization_plot["ci95_margin"],
#     marker="o",
#     linewidth=2,
#     markersize=7,
#     capsize=5,
# )
#
# axis.set_title(
#     "AWS Lambda Cold-Start Initialization Time",
#     fontsize=15,
#     fontweight="bold",
#     pad=14,
# )
#
# axis.set_xlabel(
#     "Configured Lambda memory (MB)",
#     fontsize=11,
# )
#
# axis.set_ylabel(
#     "Mean initialization duration (ms)",
#     fontsize=11,
# )
#
# axis.set_xticks(MEMORY_CONFIGURATIONS)
#
# axis.grid(
#     axis="y",
#     linestyle="--",
#     alpha=0.35,
# )
#
# axis.spines["top"].set_visible(False)
# axis.spines["right"].set_visible(False)
#
# fig.text(
#     0.5,
#     0.01,
#     "Points show sample means; error bars show 95% t-confidence intervals.",
#     ha="center",
#     fontsize=9,
# )
#
# fig.tight_layout(rect=[0, 0.04, 1, 1])
#
# plt.show()
#
# ---------------------------------------------------------
# Estimated AWS Lambda cost per 1,000 invocations
# ---------------------------------------------------------

# AWS Lambda standard x86 on-demand rates.
# These estimates exclude the AWS Free Tier.
GB_SECOND_PRICE = 0.0000166667
REQUEST_PRICE_PER_MILLION = 0.20
NUMBER_OF_INVOCATIONS = 1_000


cost_summary = (
    master_data
    .groupby(["memory_mb", "start_type"])
    .agg(
        samples=("billed_ms", "count"),
        average_billed_ms=("billed_ms", "mean"),
        average_latency_ms=("latency_ms", "mean"),
    )
    .reset_index()
)


# Convert configured memory from MB to GB.
cost_summary["configured_memory_gb"] = (
    cost_summary["memory_mb"] / 1024
)


# Convert billed duration from milliseconds to seconds.
cost_summary["average_billed_seconds"] = (
    cost_summary["average_billed_ms"] / 1000
)


# Compute usage for 1,000 invocations.
cost_summary["gb_seconds_per_1000"] = (
    cost_summary["configured_memory_gb"]
    * cost_summary["average_billed_seconds"]
    * NUMBER_OF_INVOCATIONS
)


# Duration-based compute charge.
cost_summary["compute_cost_per_1000_usd"] = (
    cost_summary["gb_seconds_per_1000"]
    * GB_SECOND_PRICE
)


# Lambda request charge.
request_cost_per_1000 = (
    NUMBER_OF_INVOCATIONS
    / 1_000_000
    * REQUEST_PRICE_PER_MILLION
)

cost_summary["request_cost_per_1000_usd"] = (
    request_cost_per_1000
)


# Total Lambda charge.
cost_summary["total_cost_per_1000_usd"] = (
    cost_summary["compute_cost_per_1000_usd"]
    + cost_summary["request_cost_per_1000_usd"]
)


# Cost of one invocation.
cost_summary["cost_per_invocation_usd"] = (
    cost_summary["total_cost_per_1000_usd"]
    / NUMBER_OF_INVOCATIONS
)


# Sort the table clearly.
cost_summary = cost_summary.sort_values(
    ["start_type", "memory_mb"]
)


columns_to_display = [
    "memory_mb",
    "start_type",
    "samples",
    "average_latency_ms",
    "average_billed_ms",
    "gb_seconds_per_1000",
    "compute_cost_per_1000_usd",
    "request_cost_per_1000_usd",
    "total_cost_per_1000_usd",
]


print("\nEstimated Lambda cost per 1,000 invocations:")
print(
    cost_summary[columns_to_display]
    .round({
        "average_latency_ms": 2,
        "average_billed_ms": 2,
        "gb_seconds_per_1000": 4,
        "compute_cost_per_1000_usd": 6,
        "request_cost_per_1000_usd": 6,
        "total_cost_per_1000_usd": 6,
    })
    .to_string(index=False)
)