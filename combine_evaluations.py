import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path("reports")
CSV_NAME = "evaluation_summary.csv"
OUTPUT = Path("reports/combined_evaluation_summary.csv")

LEVELS = [f"L{i}" for i in range(1, 8)]
LEVEL_METRICS = [
    "Total_Tasks",
    "Evaluated_Tasks",
    "Avg_Exec_Time",
    "Avg_Tokens",
    "Avg_TPS",
    "Avg_TTFT",
]
AGG_AVG_COLS = ["Avg_Exec_Time", "Avg_Tokens", "Avg_TPS", "Avg_TTFT"]
RRR_COLS = [f"{lvl}_RRR" for lvl in LEVELS]
SR_COLS = [f"{lvl}_SR" for lvl in LEVELS]
EPR_COLS = [f"{lvl}_EPR_CVR" for lvl in LEVELS]
PASSK_COLS = [f"{lvl}_pass@k" for lvl in LEVELS]

MODEL_VENDOR_MAP = {
    "kakaocorp_kanana-1.5-8b-instruct-2505": ("Kakao", "OSS"),
    "skt_A.X-4.0-Light": ("SKT", "OSS"),
    "Qwen_qwen3-8B": ("Alibaba", "OSS"),
    "gemini_gemini-2.5-pro": ("Google", "API"),
    "gemini_gemini-2.5-flash": ("Google", "API"),
    "Qwen_Qwen3-4B-Instruct-2507": ("Alibaba", "OSS"),
    "K-intelligence_Midm-2.0-Base-Instruct": ("KT", "OSS"),
    "anthropic_claude-sonnet-4-20250514": ("Anthropic", "API"),
    "azure_gpt-4.1": ("OpenAI", "API"),
    "azure_gpt-5": ("OpenAI", "API"),
    "bedrock_openai.gpt-oss-120b-1:0": ("OpenAI", "OSS"),
    "bedrock_openai.gpt-oss-20b-1:0": ("OpenAI", "OSS"),
    "bedrock_qwen.qwen3-32b-v1:0": ("Alibaba", "OSS"),
}

SPECIAL_MAP = {
    "L1_TooAcc": ("L1", "ToolAcc"),
    "L1_ArgAcc": ("L1", "ArgAcc"),
    "L1_CallEM": ("L1", "CallEM"),
    "L1_RespOK": ("L1", "RespOK"),
    "L2_SelectAcc": ("L2", "SelectAcc"),
    "L3_FSM": ("L3", "FSM"),
    "L3_PSM": ("L3", "PSM"),
    "L3_ΔSteps_norm": ("L3", "ΔSteps_norm"),
    "L3_ProvAcc": ("L3", "ProvAcc"),
    "L4_Coverage": ("L4", "Coverage"),
    "L4_SourceEPR": ("L4", "SourceEPR"),
    "L5_AdaptiveRoutingScore": ("L5", "AdaptiveRoutingScore"),
    "L5_FallbackSR": ("L5", "FallbackSR"),
    "L6_ReuseRage": ("L6", "ReuseRate"),
    "L6_RedundantCallRate": ("L6", "RedundantCallRate"),
    "L6_EffScore": ("L6", "EffScore"),
    "L7_ContextRetention": ("L7", "ContextRetention"),
    "L7_RefRecall": ("L7", "RefRecall"),
}

COLUMN_ORDER = (
    ["Model", "Vendor", "Model Type"]
    + [
        f"{lvl}_{metric}"
        for metric in LEVEL_METRICS
        for lvl in LEVELS
    ]
    + RRR_COLS
    + SR_COLS
    + EPR_COLS
    + PASSK_COLS
    + list(SPECIAL_MAP.keys())
)

def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    total_weight = weights.sum()
    if math.isclose(total_weight, 0.0):
        return float("nan")
    return (series * weights).sum() / total_weight

records = []
for csv_path in BASE_DIR.glob(f"*/{CSV_NAME}"):
    df = pd.read_csv(csv_path)
    model = csv_path.parent.name.rsplit("_", 1)[0]
    level_df = df.set_index("Level")

    vendor, model_type = MODEL_VENDOR_MAP.get(model, ("Unknown", "Unknown"))
    row = {"Model": model, "Vendor": vendor, "Model Type": model_type}

    for lvl in LEVELS:
        if lvl not in level_df.index:
            continue
        for metric in LEVEL_METRICS:
            row[f"{lvl}_{metric}"] = level_df.at[lvl, metric]

    weights = df["Evaluated_Tasks"]
    for col in AGG_AVG_COLS:
        row[col] = weighted_average(df[col], weights)

    for lvl in LEVELS:
        if lvl not in level_df.index:
            continue
        row[f"{lvl}_RRR"] = level_df.at[lvl, "RRR"]
        row[f"{lvl}_SR"] = level_df.at[lvl, "SR"]
        row[f"{lvl}_EPR_CVR"] = level_df.at[lvl, "EPR_CVR"]
        row[f"{lvl}_pass@k"] = level_df.at[lvl, "pass@k"]

    for new_name, (lvl, original) in SPECIAL_MAP.items():
        row[new_name] = (
            level_df.at[lvl, original] if lvl in level_df.index else float("nan")
        )

    records.append(row)

combined = pd.DataFrame(records)
for col in COLUMN_ORDER:
    if col not in combined:
        combined[col] = pd.NA

combined = combined[COLUMN_ORDER]
combined.to_csv(OUTPUT, index=False)
print(f"Saved merged summary to {OUTPUT}")