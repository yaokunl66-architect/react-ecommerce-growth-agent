# -*- coding: utf-8 -*-
"""AARRR 五环节分析工具（Function Calling 的「能力层」）。

这些是 ReAct Agent 的「手」——被循环按需调用，负责精确算数。
大脑（react_agent.py）负责「思考」，工具负责「行动」。

接入真实大模型时，只需给这些函数套上 LangChain 的 @tool 装饰器，
即可变成标准的 Function Calling 工具。
"""

# 环节英文键 → 中文名
STAGE_CN = {"acquisition": "拉新", "activation": "激活", "retention": "留存",
            "revenue": "收入", "referral": "裂变"}

# 增长策略知识（最终「Answer」环节合成建议用，配合 RAG 检索到的知识库）
STRATEGY_KNOWLEDGE = {
    "拉新": "优先加大 ROI 高的渠道投放，优化 CAC 高的渠道素材；尝试社交裂变降低整体获客成本。",
    "激活": "优化新手引导与落地页，缩短首单路径，设置新客首单优惠券提升激活率。",
    "留存": "找到并强化 Aha Moment，对低留存用户推送召回策略（优惠券 + 内容）。",
    "收入": "通过会员体系、满减、组合销售提升客单价与复购率，聚焦 LTV/CAC 杠杆。",
    "裂变": "设计拼团/邀请返现等玩法，降低分享门槛，同时设置防刷机制。",
}


def health_grade(name: str, value: float, ok_ge: float, watch_ge: float) -> dict:
    """三级健康评级：value >= ok_ge → OK；>= watch_ge → WATCH；否则 ALERT。"""
    if value >= ok_ge:
        return {"status": "OK", "desc": f"{name} 健康"}
    if value >= watch_ge:
        return {"status": "WATCH", "desc": f"{name} 需关注"}
    return {"status": "ALERT", "desc": f"{name} 告急"}


def skill_acquisition(channels) -> dict:
    """拉新分析：计算各渠道 CAC、ROI，输出渠道质量评分。"""
    total_spend = channels["spend"].sum()
    total_users = channels["new_users"].sum()
    overall_cac = total_spend / total_users
    overall_roi = channels["gmv"].sum() / total_spend
    best = channels.sort_values("roi", ascending=False).iloc[0]
    worst = channels.sort_values("roi").iloc[0]
    grade = health_grade("渠道整体 ROI", overall_roi, ok_ge=2.0, watch_ge=1.0)
    return {
        "skill": "acquisition_analysis",
        "overall_cac": round(float(overall_cac), 2),
        "overall_roi": round(float(overall_roi), 2),
        "total_spend": int(total_spend), "total_new_users": int(total_users),
        "best_channel": {"name": best["name"], "roi": round(float(best["roi"]), 2)},
        "worst_channel": {"name": worst["name"], "roi": round(float(worst["roi"]), 2)},
        "grade": grade["status"],
    }


def skill_activation(activation: dict) -> dict:
    """激活分析：计算激活率与首单转化率。"""
    act_rate = activation["activated"] / activation["new_users"]
    order_rate = activation["first_order"] / activation["new_users"]
    grade = health_grade("激活率", act_rate, ok_ge=0.6, watch_ge=0.5)
    return {"skill": "activation_analysis", "activation_rate": round(act_rate, 3),
            "first_order_rate": round(order_rate, 3), "grade": grade["status"]}


def skill_retention(retention: dict) -> dict:
    """留存分析：输出 D1/D7/D30 留存率。"""
    r = retention["rate"]
    grade = health_grade("次日留存", r[0], ok_ge=0.35, watch_ge=0.3)
    return {"skill": "retention_analysis",
            "d1": r[0], "d7": r[1], "d30": r[2], "grade": grade["status"]}


def skill_revenue(revenue: dict, cac: float) -> dict:
    """收入分析：计算 LTV、LTV/CAC 杠杆。"""
    arppu = revenue["arppu"]
    lifetime_days = revenue["avg_lifecycle_days"]
    ltv = arppu * (lifetime_days / 30) * (1 + revenue["repurchase_rate"])
    ltv_cac = ltv / cac if cac > 0 else 0
    grade = health_grade("LTV/CAC", ltv_cac, ok_ge=3.0, watch_ge=1.0)
    return {"skill": "revenue_analysis", "gmv": round(float(revenue["gmv"]), 0),
            "arppu": arppu, "ltv": round(ltv, 2),
            "ltv_cac": round(ltv_cac, 2), "grade": grade["status"]}


def skill_referral(referral: dict) -> dict:
    """裂变分析：计算 K 因子（病毒系数）。"""
    k = referral["share_rate"] * referral["avg_invites"] * referral["invite_conv"]
    grade = health_grade("K因子", k, ok_ge=1.0, watch_ge=0.5)
    return {"skill": "referral_analysis", "k_factor": round(k, 3), "grade": grade["status"]}


# 五个分析工具注册表（按 action 名 → 函数）
ANALYSIS_TOOLS = {
    "acquisition_analysis": skill_acquisition,
    "activation_analysis": skill_activation,
    "retention_analysis": skill_retention,
    "revenue_analysis": skill_revenue,
    "referral_analysis": skill_referral,
}
