# -*- coding: utf-8 -*-
"""AARRR 电商用户增长模拟数据生成器。"""
import numpy as np
import pandas as pd


def generate_data(seed: int = 42) -> dict:
    """生成 AARRR 五环节的电商用户增长模拟数据。

    采用「业务逻辑合理的基准值」而非纯随机，保证分析结果可解释、可复现。
    接入真实数据时，只需把本函数的返回值替换为真实数据源（CSV / 数据库）。
    """
    np.random.seed(seed)

    # 1. 拉新（Acquisition）：5 个渠道的投放数据
    channels = [
        {"name": "信息流广告", "spend": 80000, "new_users": 4000,
         "impressions": 2000000, "clicks": 62000, "gmv": 120000},
        {"name": "搜索广告", "spend": 60000, "new_users": 3000,
         "impressions": 800000, "clicks": 52000, "gmv": 180000},
        {"name": "社交裂变", "spend": 30000, "new_users": 5000,
         "impressions": 1500000, "clicks": 45000, "gmv": 90000},
        {"name": "内容营销", "spend": 40000, "new_users": 2000,
         "impressions": 3000000, "clicks": 38000, "gmv": 100000},
        {"name": "KOL 合作", "spend": 50000, "new_users": 2500,
         "impressions": 1200000, "clicks": 40000, "gmv": 75000},
    ]
    df_channels = pd.DataFrame(channels)
    df_channels["cac"] = df_channels["spend"] / df_channels["new_users"]
    df_channels["roi"] = df_channels["gmv"] / df_channels["spend"]
    df_channels["ctr"] = df_channels["clicks"] / df_channels["impressions"]

    # 2. 激活（Activation）：新用户激活 / 首单
    activation = {
        "new_users": int(df_channels["new_users"].sum()),
        "activated": 9800,
        "first_order": 7200,
    }

    # 3. 留存（Retention）：D1 / D7 / D30 留存曲线
    retention = {"day": [1, 7, 30], "rate": [0.42, 0.22, 0.12]}

    # 4. 收入（Revenue）：GMV / ARPPU / 复购率 / 生命周期
    revenue = {
        "gmv": float(df_channels["gmv"].sum()),
        "arppu": 260.0,
        "repurchase_rate": 0.28,
        "avg_lifecycle_days": 180.0,
    }

    # 5. 裂变（Referral）：分享率 / 人均邀请 / 邀请转化
    referral = {"share_rate": 0.18, "avg_invites": 3.5, "invite_conv": 0.12}

    return {
        "channels": df_channels,
        "activation": activation,
        "retention": retention,
        "revenue": revenue,
        "referral": referral,
    }
