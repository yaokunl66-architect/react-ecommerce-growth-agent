# -*- coding: utf-8 -*-
"""
ReAct 电商增长 AI Agent —— 入口。

手写 ReAct 循环（Thought → Action → Observation）+ ChromaDB RAG 知识检索
+ 多格式报告输出。核心价值：把「思考—行动—观察」的决策过程完整透明地打印出来。

运行：
    pip install -r requirements.txt
    python run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_data
from src.react_agent import ReactAgent
from src import report
from src.tools import STAGE_CN

ICON = {"OK": "[OK]", "WATCH": "[WATCH]", "ALERT": "[ALERT]"}


def main():
    print("=" * 62)
    print("  ReAct 电商增长 AI Agent（ChromaDB + RAG）")
    print("=" * 62)

    # 1. 数据生成
    data = generate_data()

    # 2. 跑 ReAct 循环（Thought → Action → Observation）
    agent = ReactAgent()
    out = agent.run(data)

    # 3. 打印完整决策轨迹（ReAct 的「可观测性」）
    print("\n  ── ReAct 决策轨迹（思考 → 行动 → 观察）──")
    for t in out["trace"]:
        print(f"\n  [Step {t['step']}]")
        print(f"    Thought     : {t['thought']}")
        print(f"    Action      : {t['action']}")
        obs = t["observation"]
        if isinstance(obs, dict):
            if t["action"] == "retrieve_knowledge":
                print(f"    Observation : 检索到 {len(obs.get('docs', []))} 条知识（通道={obs.get('method')}）")
            elif "grade" in obs:
                print(f"    Observation : {obs.get('skill', '')} → 评级 {obs['grade']}")
            else:
                print(f"    Observation : {obs}")
        else:
            print(f"    Observation : {obs}")

    # 4. 控制台摘要
    results = out["report"]
    print("\n  ── AARRR 五环节健康度 ──")
    for k, v in results.items():
        if k == "strategy":
            continue
        print(f"  {ICON.get(v['grade'], '')} {STAGE_CN[k]:>6} ({k}): {v['grade']}")

    print("\n  策略建议：")
    for s in results["strategy"]["suggestions"]:
        print(f"  * [{s['stage']}] {s['advice']}")

    print(f"\n  RAG 检索（通道={results['strategy'].get('retrieval_method')}）知识库参考：")
    for d in results["strategy"].get("knowledge", []):
        print(f"    · {d}")

    # 5. 报告输出
    os.makedirs("output", exist_ok=True)
    report.write_json(results, "output/report.json")
    report.write_txt(results, "output/report.txt")
    report.plot_charts(data)

    print("\n  报告已输出到 output/ 目录。")
    print("=" * 62)


if __name__ == "__main__":
    main()
