# -*- coding: utf-8 -*-
"""RAG 知识库引擎（ChromaDB 为核心，关键词匹配兜底）。

这是本项目的「记忆 / 领域知识」来源：Agent 用「先查再答」的方式，
从知识库检索增长打法，再结合分析结果给出策略——即 RAG（检索增强生成）。

检索策略（双通道，ChromaDB 优先）：
1. ChromaDB 语义检索：向量化 + 相似度召回（真正的 RAG 语义检索）；
2. 关键词 Jaccard 匹配：零依赖兜底，中文短文档下也稳定。

注意：ChromaDB 默认 embedding（all-MiniLM-L6-v2）为英文模型，对中文语义
召回一般，故本引擎保留关键词兜底；生产环境可换中文 embedding（见 README）。
"""

# 电商增长知识库（AARRR 方法论 + 渠道策略 + 留存打法 + 裂变玩法）
KNOWLEDGE_BASE = [
    "AARRR 模型把用户增长拆为五个环节：拉新、激活、留存、收入、自传播。",
    "拉新环节核心指标是 CAC（获客成本）和渠道 ROI，健康标准是 CAC 越低越好，LTV/CAC 大于 3 才算健康。",
    "激活环节看新用户是否完成关键动作（首单/注册后 24h 内下单），激活率低于 50% 说明新手引导或落地页有问题。",
    "留存环节看次日/7日/30日留存率，次日留存低于 30% 说明产品价值感不足，需要优化 Aha Moment。",
    "收入环节看 GMV、ARPU、LTV、客单价、复购率，LTV/CAC 是判断增长是否可持续的核心杠杆。",
    "自传播环节看 K 因子（病毒系数），K = 分享率 × 人均邀请数 × 邀请转化率，K 大于 1 才能自增长。",
    "信息流广告适合拉新放量，但 CAC 偏高，需要关注素材点击率和落地页转化率。",
    "搜索广告用户意图明确、转化率高，适合承接高购买意愿用户，ROI 通常优于信息流。",
    "社交裂变通过老带新降低 CAC，是性价比最高的拉新方式，核心是设计好激励与分享门槛。",
    "内容营销获客成本低、用户质量高，但起量慢，适合做长期品牌与信任建设。",
    "KOL 合作能快速起量，但需要评估达人粉丝与目标人群的匹配度，避免无效曝光。",
    "提升留存的关键是找到 Aha Moment（惊喜时刻），让新用户在首次使用就感受到核心价值。",
    "针对流失用户可设计召回策略：短信/Push + 优惠券组合，召回成本通常低于拉新成本。",
    "裂变玩法的核心是降低分享门槛、放大分享动机，如拼团、助力、邀请返现、阶梯奖励。",
    "裂变要防止羊毛党，设置防刷机制（设备指纹、邀请上限、奖励延迟到账）。",
]


class RagEngine:
    """知识库检索引擎。

    - ChromaDB 语义检索：主通道（向量化 + 相似度召回）。
    - 关键词 Jaccard 匹配：兜底通道（零依赖、中文短文档稳定）。

    search() 返回 (docs, method)，method ∈ {"chromadb", "keyword"}，
    便于在日志 / 报告中看到「本次检索走了哪条通道」。
    """

    def __init__(self, knowledge_base=None, collection_name: str = "ecommerce_growth"):
        self.kb = knowledge_base or KNOWLEDGE_BASE
        self._collection = None
        self._init_chroma(collection_name)

    def _init_chroma(self, collection_name: str):
        """初始化 ChromaDB 集合（失败则回退关键词匹配，保证离线可跑）。"""
        try:
            import chromadb
            client = chromadb.Client()
            coll = client.get_or_create_collection(collection_name)
            if coll.count() == 0:  # 首次运行才灌库，避免重复
                coll.upsert(
                    ids=[str(i) for i in range(len(self.kb))],
                    documents=self.kb,
                )
            self._collection = coll
        except Exception:
            self._collection = None

    def search(self, query: str, top_k: int = 3):
        """检索与 query 最相关的 top_k 条知识。返回 (docs, method)。"""
        if self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query], n_results=top_k)
                docs = res.get("documents", [[]])[0]
                if docs:
                    return list(docs), "chromadb"
            except Exception:
                pass
        return self._keyword_search(query, top_k), "keyword"

    def _keyword_search(self, query: str, top_k: int = 3):
        """Jaccard 相似度关键词匹配（兜底方案）。"""
        q_words = set(query)
        scored = []
        for doc in self.kb:
            d_words = set(doc)
            inter = len(q_words & d_words)
            union = len(q_words | d_words)
            scored.append((inter / union if union else 0, doc))
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k]]
