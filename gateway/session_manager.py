#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: gateway/session_manager.py
职责:
1. 多租户/多会话隔离 (基于 session_id)
2. 保存解析后的合同条款
3. 定期淘汰超时过期会话，释放服务器内存
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ReviewSession:
    """单个审查会话状态实体"""
    session_id: str
    last_active: float = field(default_factory=time.time)
    contract_clauses: List[Dict[str, Any]] = field(default_factory=list)  # 切分后的合同条款

    def touch(self):
        """刷新最近活跃时间戳"""
        self.last_active = time.time()


class SessionManager:
    """会话生命周期与滑动窗口治理管理器"""

    def __init__(
        self,
        session_timeout_seconds: int = 3600,
    ):
        """
        :param session_timeout_seconds: 会话过期淘汰时长（默认 1 小时）
        """
        self._sessions: Dict[str, ReviewSession] = {}
        self.session_timeout = session_timeout_seconds

    def create_session(self) -> str:
        """创建一个全新的会话"""
        self.cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        session = ReviewSession(session_id=session_id)
        self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[ReviewSession]:
        """获取并激活会话"""
        session = self._sessions.get(session_id)
        if session:
            # 检查是否过期
            if time.time() - session.last_active > self.session_timeout:
                self.delete_session(session_id)
                return None
            session.touch()
        return session

    def store_clauses(self, session_id: str, clauses: List[Dict[str, Any]]) -> bool:
        """将切分好的合同条款挂载至会话中"""
        session = self.get_session(session_id)
        if not session:
            return False
        session.contract_clauses = clauses
        return True

    def delete_session(self, session_id: str):
        """显式销毁会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def cleanup_expired_sessions(self):
        """清理所有超时的不活跃会话"""
        now = time.time()
        expired_ids = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.session_timeout
        ]
        for sid in expired_ids:
            del self._sessions[sid]


# 全局单例管理器
session_manager = SessionManager()
