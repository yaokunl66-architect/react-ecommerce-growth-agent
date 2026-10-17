# -*- coding: utf-8 -*-
"""报告输出：JSON / TXT / PNG 图表。"""
import json
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"
ICON = {"OK": "🟢", "WATCH": "🟡", "ALERT": "🔴"}


def write_json(report: dict, path: str):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    payload = {"title": "ReAct 电商增长 AI Agent - AARRR 分析报告",
               "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "summary": {k: v["grade"] for k, v in report.items() if k != "strategy"},
               "detail": report}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)


def write_txt(report: dict, path: str):
    lines = ["=" * 60, "  ReAct 电商增长 AI Agent - AARRR 分析报告",
             "=" * 60, f"  生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    lines.append("  --- AARRR 五环节健康度 ---")
    for k, v in report.items():
        if k == "strategy":
            continue
        lines.append(f"  {ICON.get(v['grade'], '')} {k:>12}: {v['grade']}")
    lines.append("")
    lines.append("  --- 策略建议 ---")
    for s in report["strategy"]["suggestions"]:
        lines.append(f"  * [{s['stage']}] {s['advice']}")
    lines.append("")
    lines.append("  --- RAG 检索到的知识库参考 ---")
    lines.append(f"  检索通道: {report['strategy'].get('retrieval_method', 'keyword')}")
    for d in report["strategy"].get("knowledge", []):
        lines.append(f"  · {d}")
    lines.append("")
    lines.append("=" * 60)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_charts(data: dict, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    ch = data["channels"]

    # 1. AARRR 漏斗图
    stages = ["拉新", "激活", "首单", "留存(D7)", "裂变参与"]
    vals = [int(ch["new_users"].sum()), data["activation"]["activated"],
            data["activation"]["first_order"],
            int(data["activation"]["first_order"] * data["retention"]["rate"][1]),
            int(data["activation"]["first_order"] * data["referral"]["share_rate"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(stages)), vals,
            color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"])
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel("用户数")
    ax.set_title("AARRR 用户增长漏斗")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aarrr_funnel.png", dpi=150)
    plt.close()

    # 2. 留存曲线
    ret = data["retention"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ret["day"], ret["rate"], marker="o", color="#e74c3c", linewidth=2)
    for x, y in zip(ret["day"], ret["rate"]):
        ax.text(x, y + 0.01, f"{y:.0%}", ha="center")
    ax.set_xlabel("天数")
    ax.set_ylabel("留存率")
    ax.set_title("用户留存曲线（D1/D7/D30）")
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/retention_curve.png", dpi=150)
    plt.close()

    # 3. 渠道对比图
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(ch))
    ax.bar(x, ch["roi"], color="#3498db", label="ROI")
    ax2 = ax.twinx()
    ax2.plot(x, ch["cac"], marker="o", color="#e74c3c", label="CAC", linewidth=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ch["name"])
    ax.set_ylabel("ROI")
    ax2.set_ylabel("CAC (元)")
    ax.set_title("渠道 ROI vs CAC 对比")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/channel_compare.png", dpi=150)
    plt.close()
