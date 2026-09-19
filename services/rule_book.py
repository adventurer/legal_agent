"""Public facade for the enterprise review rule book.

The historical ``EnterpriseRuleBook`` import remains the compatibility
boundary while persistence and formatting live in focused modules.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from services.rule_book_formatting import search_rules
    from services.rule_book_storage import RuleBookStore
except ImportError:  # Support ``python services/rule_book.py`` as before.
    from rule_book_formatting import search_rules
    from rule_book_storage import RuleBookStore

try:
    from configs.config import RULE_BOOK_DB_PATH
except ImportError:
    RULE_BOOK_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rule_book.db"


class EnterpriseRuleBook:
    """Enterprise rule-book facade retaining the original public API."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or RULE_BOOK_DB_PATH).resolve()
        self._store = RuleBookStore(self.db_path)

    def _get_connection(self):
        return self._store.connection()

    def search_rules(self, query: str, top_k: int = 2) -> str:
        return search_rules(self._store.all(), query, top_k)

    def upsert_rule(
        self, topic: str, keywords: str, risk_level: str,
        standard_requirement: str, forbidden_pattern: Optional[str] = None,
        recommended_clause: Optional[str] = None,
    ) -> bool:
        return self._store.upsert(
            topic, keywords, risk_level, standard_requirement,
            forbidden_pattern, recommended_clause,
        )

    def list_all_rules(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._store.all()]

    def delete_rule_by_topic(self, topic: str) -> bool:
        return self._store.delete_topic(topic)


if __name__ == "__main__":
    rule_book = EnterpriseRuleBook()
    print(f"[+] 数据库初始化就绪: {rule_book.db_path}")
    print(f"[+] 当前已收录规则条数: {len(rule_book.list_all_rules())}")
    print(rule_book.search_rules("违约金"))
