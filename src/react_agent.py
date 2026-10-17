# -*- coding: utf-8 -*-
"""ReAct Agent 核心循环：Thought → Action → Observation → ... → Answer。

ReAct = Reasoning（推理）+ Acting（行动）。Agent 在「思考」与「行动」之间交替：
每一步先思考（Thought）该干什么，再执行（Action）调用工具，观察（Observation）
工具返回结果，据此继续思考，直到完成任务给出最终答案（Answer）。

本文件包含：
1. ReactAgent —— 手写的 ReAct 循环（不依赖 LangChain，把循环过程透明化）；
2. MockReactBrain —— 确定性大脑（零 Key、零网络、可复现，用于离线教学演示）；
3. LLMReactBrain —— 可插拔真实大模型大脑（DeepSeek / OpenAI 兼容，可选）。
"""
from .rag import RagEngine
from .tools import ANALYSIS_TOOLS, STAGE_CN, STRATEGY_KNOWLEDGE

# 各环节「关注点关键词」——ReAct 在 retrieve_knowledge 环节据此构建检索 query
STAGE_QUERY = {
    "acquisition": "如何降低获客成本、提升渠道 ROI",
    "activation": "如何提升新用户激活率与首单转化",
    "retention": "如何提升用户留存率、优化 Aha Moment",
    "revenue": "如何提升 LTV 与复购率、放大 LTV/CAC",
    "referral": "如何提升裂变 K 因子、设计裂变玩法",
}


class ReactAgent:
    """ReAct 循环执行器。

    用法：
        agent = ReactAgent()          # 默认 MockReactBrain + ChromaDB RAG
        out = agent.run(data)         # out = {"report": {...}, "trace": [...]}
    """

    def __init__(self, brain=None, rag=None, max_steps: int = 12):
        self.rag = rag or RagEngine()
        self.brain = brain or MockReactBrain(self.rag)
        self.max_steps = max_steps
        self.tools = dict(ANALYSIS_TOOLS)
        self.tools["retrieve_knowledge"] = self._retrieve_knowledge

    def _retrieve_knowledge(self, query: str, top_k: int = 3) -> dict:
        """retrieve_knowledge 工具：从 ChromaDB 知识库检索增长打法（RAG）。"""
        docs, method = self.rag.search(query, top_k)
        return {"query": query, "docs": docs, "method": method}

    def run(self, data: dict) -> dict:
        """跑一轮 ReAct 循环，返回最终报告 + 完整决策轨迹。"""
        trace = []          # 每一步的 {thought, action, observation}
        memory = {}         # 累积的观察结果（后续步骤可引用，如收入需要拉新的 CAC）

        for step in range(1, self.max_steps + 1):
            decision = self.brain.think(data, memory, trace)
            action = decision["action"]

            if action == "finish":
                answer = self.brain.answer(data, memory, trace)
                trace.append({"step": step, "thought": decision["thought"],
                              "action": "finish", "observation": answer["summary"]})
                return {"report": answer["report"], "trace": trace}

            # 执行 Action → 得到 Observation
            tool = self.tools[action]
            observation = tool(**decision["tool_input"])
            memory[action] = observation
            trace.append({"step": step, "thought": decision["thought"],
                          "action": action, "observation": observation})

        # 兜底：超过 max_steps 强制收尾
        answer = self.brain.answer(data, memory, trace)
        return {"report": answer["report"], "trace": trace}


class MockReactBrain:
    """确定性大脑：按固定计划「思考 → 行动」，产出显式 Thought/Action 轨迹。

    计划（5 步分析 + 1 步 RAG 检索 + 收尾）：
      拉新分析 → 激活分析 → 留存分析 → 收入分析 → 裂变分析 → 知识检索 → Answer
    """

    PLAN = [
        ("acquisition_analysis", "先分析拉新效率：算整体 CAC 与渠道 ROI，找最佳/最差渠道"),
        ("activation_analysis", "再分析激活：算激活率与首单转化率"),
        ("retention_analysis", "接着分析留存：看 D1/D7/D30 留存曲线"),
        ("revenue_analysis", "然后结合 CAC 分析收入杠杆：算 LTV 与 LTV/CAC"),
        ("referral_analysis", "最后分析裂变：算 K 因子（病毒系数）"),
        ("retrieve_knowledge", "针对告急/关注环节，从知识库检索增长打法（RAG）"),
    ]

    def __init__(self, rag: RagEngine):
        self.rag = rag

    def think(self, data: dict, memory: dict, trace: list) -> dict:
        done = {t["action"] for t in trace}
        for action, thought in self.PLAN:
            if action not in done:
                return {"thought": thought, "action": action,
                        "tool_input": self._tool_input(action, data, memory)}
        return {"thought": "五个环节分析完毕，综合所有观察与知识库，产出最终答案",
                "action": "finish", "tool_input": {}}

    def _tool_input(self, action: str, data: dict, memory: dict) -> dict:
        """为每个 action 准备工具入参（含跨步骤依赖：收入需要拉新的 CAC）。"""
        if action == "acquisition_analysis":
            return {"channels": data["channels"]}
        if action == "activation_analysis":
            return {"activation": data["activation"]}
        if action == "retention_analysis":
            return {"retention": data["retention"]}
        if action == "revenue_analysis":
            cac = memory["acquisition_analysis"]["overall_cac"]
            return {"revenue": data["revenue"], "cac": cac}
        if action == "referral_analysis":
            return {"referral": data["referral"]}
        if action == "retrieve_knowledge":
            return {"query": self._build_query(memory)}
        return {}

    def _build_query(self, memory: dict) -> str:
        """根据告急/关注环节拼出检索 query（让 RAG 更有针对性）。"""
        problems = []
        for stage in ["acquisition", "activation", "retention", "revenue", "referral"]:
            obs = memory.get(f"{stage}_analysis", {})
            if obs.get("grade") in ("ALERT", "WATCH"):
                problems.append(STAGE_QUERY[stage])
        return "；".join(problems) if problems else "电商用户增长增长策略"

    def answer(self, data: dict, memory: dict, trace: list) -> dict:
        """汇总所有观察 + RAG 检索到的知识，产出最终报告。"""
        results = {}
        for stage in ["acquisition", "activation", "retention", "revenue", "referral"]:
            results[stage] = memory[f"{stage}_analysis"]

        rag = memory.get("retrieve_knowledge", {"docs": [], "method": "keyword"})

        # 策略建议：针对 ALERT/WATCH 环节给出动作，并附上知识库参考
        suggestions = []
        for stage, r in results.items():
            if r["grade"] in ("ALERT", "WATCH"):
                cn = STAGE_CN[stage]
                suggestions.append({"stage": cn, "advice": STRATEGY_KNOWLEDGE[cn]})
        if not suggestions:
            suggestions.append({"stage": "整体", "advice": "各环节均健康，可加大整体投放规模，追求规模增长。"})

        results["strategy"] = {
            "suggestions": suggestions,
            "knowledge": rag.get("docs", []),
            "retrieval_method": rag.get("method", "keyword"),
        }
        summary = "、".join(f"{STAGE_CN[k]}={v['grade']}" for k, v in results.items() if k != "strategy")
        return {"report": results, "summary": summary}


class LLMReactBrain(MockReactBrain):
    """真实大模型大脑（可选）：由 LLM 生成 Thought/Action，而非固定计划。

    用法：
        pip install langchain-openai
        brain = LLMReactBrain(rag, api_key=..., base_url=..., model="deepseek-chat")
        agent = ReactAgent(brain=brain, rag=rag)
    """

    def __init__(self, rag: RagEngine, api_key: str = "", base_url: str = "",
                 model: str = "deepseek-chat"):
        super().__init__(rag)
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        import os
        key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("真实大模型需要 API Key：请传入 api_key 或设置环境变量。")
        self._llm = ChatOpenAI(model=model, api_key=key, base_url=base_url, temperature=0)
        self._msgs = (SystemMessage, HumanMessage)

    def think(self, data: dict, memory: dict, trace: list) -> dict:
        # 简化实现：真实模式下把「下一步该调哪个工具」交给 LLM（此处保留 Mock 兜底）
        # 生产建议：让 LLM 返回 JSON {thought, action, tool_input} 并用 json 解析
        return super().think(data, memory, trace)
