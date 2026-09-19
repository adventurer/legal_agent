#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: services/rule_book.py
职责:
1. 基于 SQLite 实现企业自编法典与历史审查偏好规则的持久化
2. 支持规则的新增、更新（同主题/规则去重覆盖）、删除与全量导出
3. 提供按关键词/条款主题的高效加权检索算法，输出带红线等级的审查指引
4. 内置企业常见合规红线预置数据，开箱即用
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

# 项目根目录对齐
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 从全局配置中读取数据库路径
try:
    from configs.config import RULE_BOOK_DB_PATH
except ImportError:
    RULE_BOOK_DB_PATH = ROOT_DIR / "data" / "rule_book.db"


class EnterpriseRuleBook:
    """企业自编法典与审查偏好记忆库"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or RULE_BOOK_DB_PATH).resolve()
        # 确保数据存储目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._ensure_preset_rules()

    def _get_connection(self) -> sqlite3.Connection:
        """获取 SQLite 数据库连接并配置行映射"""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """初始化数据库表结构与索引"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL UNIQUE,          -- 条款主题（唯一键，用于覆盖更新）
                    keywords TEXT NOT NULL,              -- 匹配关键词（逗号分隔）
                    risk_level TEXT NOT NULL,            -- 企业红线级别: High / Medium / Low
                    standard_requirement TEXT NOT NULL,  -- 企业标准控制红线/偏好要求
                    forbidden_pattern TEXT,              -- 严厉禁止的表述或条款模式
                    recommended_clause TEXT,             -- 推荐替换的标准示范条款
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 创建关键词与主题索引，加速匹配
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON review_rules(topic)")
            conn.commit()

    def _ensure_preset_rules(self):
        """如果数据库为空，初始化注入企业核心通用审查红线偏好"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM review_rules")
            count = cursor.fetchone()[0]

            if count == 0:
                preset_rules = [
                    (
                        "逾期违约金比例",
                        "违约金,逾期,按日,百分之,滞纳金",
                        "High",
                        "我司对外交货或付款违约金每日不得超过合同总金额的万分之五（0.05%），严禁约定日 1% 或 5% 以上的高额惩罚性违约金。",
                        "按日支付 1% 以上违约金；每日支付合同总额 5% 违约金",
                        "乙方逾期交货的，每日应按逾期未交货物价值的万分之五向甲方支付违约金，违约金总额最高不超过逾期货物总价值的 5%。",
                    ),
                    (
                        "付款条件与结算周期",
                        "付款,验收,尾款,工作日,视资金情况",
                        "High",
                        "严禁接受带有主观随意性的付款条件（如‘视甲方资金充裕情况付款’）。尾款支付周期自验收合格起不得超过 30 个工作日。",
                        "视资金充裕情况安排支付；无确定期限的付款条款；超过 90 个工作日的过长付款周期",
                        "甲方在验收合格并收到乙方开具的增值税专用发票后 15 个工作日内，向乙方支付剩余 30% 尾款。",
                    ),
                    (
                        "争议管辖与仲裁机构",
                        "争议,管辖,仲裁,单方指定,法院",
                        "High",
                        "严禁接受单方指定仲裁员、单方确定管辖地或非正规民间仲裁。原则上必须约定我方所在地人民法院管辖，或正规仲裁委员会（如北仲、贸仲、深国仲）。",
                        "由对方单方指定独任仲裁员在其办公室内仲裁；剥夺合法诉讼救济权",
                        "因本合同引起的或与本合同有关的任何争议，双方应友好协商解决；协商不成的，任何一方均可向甲方住所地有管辖权的人民法院提起诉讼。",
                    ),
                    (
                        "知识产权归属",
                        "知识产权,专利,著作权,成果归属",
                        "Medium",
                        "合作开发或委托开发成果的知识产权，必须明确归我方所有或双方共有，严禁我方出资但知识产权全权归对方所有。",
                        "成果知识产权全部归乙方所有",
                        "在本合同履行过程中由双方合作或乙方基于甲方需求开发形成的全部知识产权，均独家归甲方所有。",
                    ),
                ]
                cursor.executemany("""
                    INSERT INTO review_rules (
                        topic, keywords, risk_level, standard_requirement, forbidden_pattern, recommended_clause
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, preset_rules)
                conn.commit()

    def upsert_rule(
        self,
        topic: str,
        keywords: str,
        risk_level: str,
        standard_requirement: str,
        forbidden_pattern: Optional[str] = None,
        recommended_clause: Optional[str] = None,
    ) -> bool:
        """
        新增或更新审查偏好规则（以 topic 作为唯一键，实现覆盖更新）
        """
        sql = """
            INSERT INTO review_rules (
                topic, keywords, risk_level, standard_requirement, forbidden_pattern, recommended_clause, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(topic) DO UPDATE SET
                keywords = excluded.keywords,
                risk_level = excluded.risk_level,
                standard_requirement = excluded.standard_requirement,
                forbidden_pattern = excluded.forbidden_pattern,
                recommended_clause = excluded.recommended_clause,
                updated_at = CURRENT_TIMESTAMP
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    topic.strip(),
                    keywords.strip(),
                    risk_level.strip(),
                    standard_requirement.strip(),
                    forbidden_pattern.strip() if forbidden_pattern else "",
                    recommended_clause.strip() if recommended_clause else "",
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"[错误] 自编法典保存规则失败: {e}")
            return False

    def search_rules(self, query: str, top_k: int = 2) -> str:
        """
        为 Agent Tool 提供关键词检索能力 (匹配 topic 与 keywords)
        :param query: 用户输入或 Agent 提取的审查关键词（如 '违约金'、'仲裁'、'付款'）
        :param top_k: 返回最相关规则条数
        :return: 格式化的法典指引字符串
        """
        if not query or not query.strip():
            return "未提供有效的查询关键词。"

        search_tokens = [token.strip() for token in query.replace(",", " ").replace("，", " ").split() if token.strip()]
        if not search_tokens:
            search_tokens = [query.strip()]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM review_rules")
            rows = cursor.fetchall()

        if not rows:
            return "企业自编法典库当前为空。"

        scored_rules = []
        for row in rows:
            score = 0
            topic = row["topic"]
            keywords = row["keywords"]
            requirement = row["standard_requirement"]
            forbidden = row["forbidden_pattern"] or ""

            # 评分加权：主题命中权重最高，关键词次之，内容文本命中兜底
            for token in search_tokens:
                if token in topic:
                    score += 5
                if token in keywords:
                    score += 3
                if token in requirement:
                    score += 1
                if token in forbidden:
                    score += 1

            if score > 0:
                scored_rules.append((score, row))

        if not scored_rules:
            return f"《企业自编法典》中暂未收录针对【{query}】的特殊禁止性规则与审查偏好。"

        # 按得分从高到低排序，截取前 top_k
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        selected = scored_rules[:top_k]

        results = []
        for idx, (_, rule) in enumerate(selected, 1):
            rule_text = (
                f"【自编法典规则 {idx}】主题：{rule['topic']} (控制红线级别: {rule['risk_level']})\n"
                f"- 企业控制要求: {rule['standard_requirement']}\n"
            )
            if rule["forbidden_pattern"]:
                rule_text += f"- 严禁模式: {rule['forbidden_pattern']}\n"
            if rule["recommended_clause"]:
                rule_text += f"- 推荐合规范本: {rule['recommended_clause']}"
            results.append(rule_text.strip())

        return "\n\n".join(results)

    def list_all_rules(self) -> List[Dict[str, Any]]:
        """获取法典中全部规则清单"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM review_rules ORDER BY id ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_rule_by_topic(self, topic: str) -> bool:
        """根据主题删除单条规则"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM review_rules WHERE topic = ?", (topic.strip(),))
            conn.commit()
            return cursor.rowcount > 0


# ==================== 单元测试与效果验证 ====================
if __name__ == "__main__":
    print("=" * 60)
    print(" 🚀 EnterpriseRuleBook 自编法典服务启动测试")
    print("=" * 60)

    # 1. 实例化
    rule_book = EnterpriseRuleBook()
    print(f"[+] 数据库初始化就绪: {rule_book.db_path}")

    # 2. 列出已加载规则
    all_rules = rule_book.list_all_rules()
    print(f"[+] 当前已收录规则条数: {len(all_rules)}")
    for r in all_rules:
        print(f"  - [{r['risk_level']}] {r['topic']} (关键词: {r['keywords']})")

    # 3. 模拟 Agent 工具检索测试
    print("\n" + "=" * 60)
    print(">>> 模拟检索 1: 查询 '违约金'")
    result_1 = rule_book.search_rules("违约金")
    print(result_1)

    print("\n" + "=" * 60)
    print(">>> 模拟检索 2: 查询 '仲裁管辖'")
    result_2 = rule_book.search_rules("单方仲裁")
    print(result_2)

    print("\n" + "=" * 60)
    print(">>> 模拟检索 3: 查询不存在的主题 '商业秘密保密金'")
    result_3 = rule_book.search_rules("商业秘密保密金")
    print(result_3)