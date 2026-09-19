"""SQLite persistence for the enterprise review rule book."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


PRESET_RULES = [
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


class RuleBookStore:
    """Owns the SQLite schema and CRUD operations, not search presentation."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._ensure_preset_rules()

    def connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS review_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL UNIQUE,
                    keywords TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    standard_requirement TEXT NOT NULL,
                    forbidden_pattern TEXT,
                    recommended_clause TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON review_rules(topic)")
            conn.commit()

    def _ensure_preset_rules(self) -> None:
        with self.connection() as conn:
            if conn.execute("SELECT COUNT(*) FROM review_rules").fetchone()[0]:
                return
            conn.executemany(
                """INSERT INTO review_rules
                   (topic, keywords, risk_level, standard_requirement,
                    forbidden_pattern, recommended_clause)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                PRESET_RULES,
            )
            conn.commit()

    def upsert(
        self, topic: str, keywords: str, risk_level: str,
        standard_requirement: str, forbidden_pattern: Optional[str],
        recommended_clause: Optional[str],
    ) -> bool:
        try:
            with self.connection() as conn:
                conn.execute(
                    """INSERT INTO review_rules
                       (topic, keywords, risk_level, standard_requirement,
                        forbidden_pattern, recommended_clause, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(topic) DO UPDATE SET
                         keywords=excluded.keywords, risk_level=excluded.risk_level,
                         standard_requirement=excluded.standard_requirement,
                         forbidden_pattern=excluded.forbidden_pattern,
                         recommended_clause=excluded.recommended_clause,
                         updated_at=CURRENT_TIMESTAMP""",
                    (
                        topic.strip(), keywords.strip(), risk_level.strip(),
                        standard_requirement.strip(),
                        forbidden_pattern.strip() if forbidden_pattern else "",
                        recommended_clause.strip() if recommended_clause else "",
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            print(f"[错误] 自编法典保存规则失败: {exc}")
            return False

    def all(self) -> List[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM review_rules ORDER BY id ASC").fetchall()

    def delete_topic(self, topic: str) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM review_rules WHERE topic = ?", (topic.strip(),))
            conn.commit()
            return cursor.rowcount > 0
