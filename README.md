# ReAct 电商增长 AI Agent（ChromaDB + RAG）

**中文** | [English](README.en.md)

> **一个手写的 ReAct 循环（Thought → Action → Observation）+ ChromaDB 向量知识库，**
> **对电商 AARRR 五环节做诊断，并基于 RAG 检索出的增长打法给出策略建议。**

这是「单 Agent 范式」的完整可运行项目：用**手写 ReAct 循环**（不依赖 LangChain）
把 Agent 的「思考 → 行动 → 观察 → 再思考」过程完整透明地暴露出来，并且把
**ChromaDB + RAG** 作为知识检索的核心（而非可选）。

---

## 一、这是什么

一个电商用户增长分析 Agent。输入（或自动生成）用户增长数据，Agent 在 ReAct
循环里一步步「思考该算什么 → 调用工具算 → 观察结果 → 继续思考」，最后基于
ChromaDB 知识库检索出的打法，输出：

- **AARRR 五环节健康诊断** — 拉新 / 激活 / 留存 / 收入 / 裂变 三级评级（ALERT/WATCH/OK）
- **核心增长指标** — CAC、ROI、LTV、LTV/CAC、K 因子、留存率
- **增长策略建议** — 从 ChromaDB 知识库检索打法，给出可落地动作
- **完整决策轨迹** — 每一步的 Thought / Action / Observation 全程打印
- **多格式报告** — JSON / TXT / 可视化图表（漏斗图、留存曲线、渠道对比）

---

## 二、快速开始

```bash
pip install -r requirements.txt
python run.py
```

零 API Key 即可跑通（确定性大脑 + ChromaDB 本地向量库）。

> 首次运行 ChromaDB 会下载默认 embedding 模型（`all-MiniLM-L6-v2`，约 90MB）。
> 若离线无法下载，引擎会自动回退到「关键词 Jaccard 匹配」，程序照常运行。

---

## 三、ReAct 循环是怎么跑的

```
用户问题：诊断电商增长健康度
  Thought: 先分析拉新效率（CAC/ROI）
  Action: acquisition_analysis
  Observation: 整体 CAC=15.76, ROI=2.17 → OK
  Thought: 再分析激活（激活率/首单）
  Action: activation_analysis
  Observation: 激活率=0.594 → WATCH
  ...（留存 / 收入 / 裂变依次分析）...
  Thought: 针对告急/关注环节，从知识库检索增长打法（RAG）
  Action: retrieve_knowledge
  Observation: 检索到 3 条知识（通道=chromadb）
  Thought: 综合所有观察 + 知识库，产出最终答案
  Action: finish
```

**关键点**：这不是「一次性算完」，而是**显式的多轮思考-行动循环**——每一步都有
Thought（为什么这么做）、Action（调哪个工具）、Observation（得到什么结果）。
这正对应课程里讲的 ReAct = Reasoning + Acting。

---

## 四、RAG + ChromaDB 的用法

`src/rag.py` 里有一个 15 条电商增长知识库，写入 ChromaDB 后按语义检索：

```python
from src.rag import RagEngine

rag = RagEngine()                        # 默认 ChromaDB（失败自动回退关键词）
docs, method = rag.search("如何提升留存率", top_k=3)
# docs  = ["留存环节看次日/7日/30日留存率...", ...]
# method = "chromadb"（或 "keyword"）
```

在 ReAct 循环里，`retrieve_knowledge` 工具就是调用它：Agent 先算出哪些环节告急，
再据此拼出 query 去知识库检索，最后把检索到的打法带进最终策略——这就是
**Agentic RAG**（检索动作由 Agent 自主决定，而非写死）。

---

## 五、项目结构

```
react-ecommerce-growth-agent/
├── run.py                  ← 入口：跑 ReAct 循环 + 打印轨迹 + 出报告
├── requirements.txt
├── README.md               ← 中文说明
├── README.en.md            ← English README
└── src/
    ├── data_generator.py   ← AARRR 模拟数据生成
    ├── rag.py              ← ChromaDB + RAG 知识库（核心）
    ├── tools.py            ← 5 个分析工具（Function Calling 能力层）
    ├── react_agent.py      ← 手写 ReAct 循环 + 确定性大脑 / 可插拔 LLM 大脑
    └── report.py           ← JSON/TXT/PNG 报告输出
```

---

## 六、接入真实大模型（可选）

`react_agent.py` 里的 `LLMReactBrain` 是可插拔大脑，把「下一步该调哪个工具」
交给 DeepSeek / OpenAI 兼容接口生成：

```bash
pip install langchain-openai
```

```python
from src.react_agent import ReactAgent, LLMReactBrain
from src.rag import RagEngine

rag = RagEngine()
brain = LLMReactBrain(rag, api_key="你的key", base_url="https://api.deepseek.com/v1")
agent = ReactAgent(brain=brain, rag=rag)
```

---

## 七、关于中文 Embedding 的说明

ChromaDB 默认 embedding（`all-MiniLM-L6-v2`）是英文模型，对中文语义召回一般，
所以本项目保留「关键词匹配」作为兜底通道（`search` 会返回实际走了哪条通道）。
生产环境建议换成中文 embedding 模型（如 `BAAI/bge-small-zh`），做法：

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction("BAAI/bge-small-zh")
client = chromadb.Client()
coll = client.get_or_create_collection("ecommerce_growth", embedding_function=ef)
```

---

## License

MIT
