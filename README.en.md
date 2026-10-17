# ReAct E-Commerce Growth AI Agent (ChromaDB + RAG)

[中文](README.md) | **English**

> **A hand-written ReAct loop (Thought → Action → Observation) plus a ChromaDB vector knowledge base.**
> **It diagnoses the five AARRR stages of an e-commerce business and recommends strategies based on growth playbooks retrieved through RAG.**

This is a complete, runnable single-agent project. It uses a **hand-written ReAct loop** (no LangChain dependency) so that the agent's "think → act → observe → think again" process is fully transparent, and it makes **ChromaDB + RAG** the core of knowledge retrieval rather than an optional add-on.

---

## 1. What It Does

An e-commerce user-growth analysis agent. Given growth data (supplied or auto-generated), the agent works through a ReAct loop: it decides what to compute, calls a tool, observes the result, and keeps reasoning. At the end, it uses playbooks retrieved from the ChromaDB knowledge base to produce:

- **AARRR health diagnosis:** a three-level rating (ALERT / WATCH / OK) for acquisition, activation, retention, revenue, and referral.
- **Core growth metrics:** CAC, ROI, LTV, LTV/CAC, K-factor, and retention rates.
- **Growth strategy suggestions:** actionable moves drawn from playbooks retrieved from the ChromaDB knowledge base.
- **Full decision trace:** every Thought / Action / Observation step is printed.
- **Multi-format reports:** JSON, TXT, and charts (funnel, retention curve, channel comparison).

---

## 2. Quick Start

```bash
pip install -r requirements.txt
python run.py
```

It runs with zero API keys (a deterministic brain plus a local ChromaDB vector store).

> On first run, ChromaDB downloads its default embedding model (`all-MiniLM-L6-v2`, about 90 MB). If you are offline and it cannot download, the engine falls back to keyword Jaccard matching and the program still runs.

---

## 3. How the ReAct Loop Works

```
User question: diagnose the growth health of the e-commerce business
  Thought: First analyze acquisition efficiency (CAC / ROI)
  Action: acquisition_analysis
  Observation: Overall CAC=15.76, ROI=2.17 → OK
  Thought: Next, analyze activation (activation rate / first order)
  Action: activation_analysis
  Observation: Activation rate=0.594 → WATCH
  ... (retention / revenue / referral analyzed in turn) ...
  Thought: For the ALERT/WATCH stages, retrieve growth playbooks from the knowledge base (RAG)
  Action: retrieve_knowledge
  Observation: Retrieved 3 knowledge items (channel=chromadb)
  Thought: Combine all observations and the knowledge base into the final answer
  Action: finish
```

**Key point:** this is not a one-shot calculation. It is an explicit multi-round think-act loop in which every step has a Thought (why), an Action (which tool), and an Observation (what came back). That is what ReAct, Reasoning + Acting, means.

---

## 4. Using RAG with ChromaDB

`src/rag.py` holds a 15-item e-commerce growth knowledge base, written into ChromaDB and searched semantically:

```python
from src.rag import RagEngine

rag = RagEngine()                        # ChromaDB by default (falls back to keywords on failure)
docs, method = rag.search("How to improve retention", top_k=3)
# docs   = ["Retention is judged by day-1 / day-7 / day-30 retention rates...", ...]
# method = "chromadb" (or "keyword")
```

Inside the ReAct loop, the `retrieve_knowledge` tool calls this engine. The agent first works out which stages are in trouble, builds a query from them, retrieves from the knowledge base, and carries the retrieved playbooks into the final strategy. This is **Agentic RAG**: the agent itself decides when to retrieve, instead of retrieval being hard-coded.

---

## 5. Project Structure

```
react-ecommerce-growth-agent/
├── run.py                  ← Entry point: run the ReAct loop, print the trace, write reports
├── requirements.txt
├── README.md               ← Chinese README
├── README.en.md            ← This document
└── src/
    ├── data_generator.py   ← Simulated AARRR data generation
    ├── rag.py              ← ChromaDB + RAG knowledge base (core)
    ├── tools.py            ← 5 analysis tools (the function-calling layer)
    ├── react_agent.py      ← Hand-written ReAct loop + deterministic brain / pluggable LLM brain
    └── report.py           ← JSON / TXT / PNG report output
```

---

## 6. Plugging in a Real LLM (Optional)

`LLMReactBrain` in `react_agent.py` is a pluggable brain that hands the decision "which tool to call next" to a DeepSeek or OpenAI-compatible endpoint:

```bash
pip install langchain-openai
```

```python
from src.react_agent import ReactAgent, LLMReactBrain
from src.rag import RagEngine

rag = RagEngine()
brain = LLMReactBrain(rag, api_key="your_api_key", base_url="https://api.deepseek.com/v1")
agent = ReactAgent(brain=brain, rag=rag)
```

---

## 7. A Note on Chinese Embeddings

ChromaDB's default embedding (`all-MiniLM-L6-v2`) is an English model, so semantic recall on Chinese text is only fair. This project therefore keeps keyword matching as a fallback channel (`search` returns which channel was actually used). In production, switch to a Chinese embedding model such as `BAAI/bge-small-zh`:

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
