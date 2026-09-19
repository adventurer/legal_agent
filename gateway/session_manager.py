#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: gateway/session_manager.py
职责:
1. 多租户/多会话隔离 (基于 session_id)
2. 维护对话历史上下文与审查中状态
3. 实现滑动窗口机制，限制单会话最大历史轮次，防止超出模型 4096 Token 上限
4. 定期淘汰超时过期会话，释放服务器内存
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ReviewSession:
    """单个审查会话状态实体"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    contract_clauses: List[Dict[str, Any]] = field(default_factory=list)  # 切分后的合同条款
    history_messages: List[Dict[str, str]] = field(default_factory=list)  # 对话历史
    metadata: Dict[str, Any] = field(default_factory=dict)                # 扩展字段

    def touch(self):
        """刷新最近活跃时间戳"""
        self.last_active = time.time()


class SessionManager:
    """会话生命周期与滑动窗口治理管理器"""

    def __init__(self, max_history_turns: int = 5, session_timeout_seconds: int = 3600):
        """
        :param max_history_turns: 滑动窗口保留的最大交互轮次（每轮包含 assistant 与 user）
        :param session_timeout_seconds: 会话过期淘汰时长（默认 1 小时）
        """
        self._sessions: Dict[str, ReviewSession] = {}
        self.max_history_turns = max_history_turns
        self.session_timeout = session_timeout_seconds

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """创建一个全新的会话"""
        self.cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        session = ReviewSession(
            session_id=session_id,
            metadata=metadata or {},
        )
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

    def append_message(self, session_id: str, role: str, content: str) -> bool:
        """追加单条消息并执行滑动窗口裁剪"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.history_messages.append({"role": role, "content": content})
        self._apply_sliding_window(session)
        return True

    def get_context_messages(self, session_id: str, system_prompt: str) -> List[Dict[str, str]]:
        """获取拼装好 system prompt 的完整上下文消息链条"""
        session = self.get_session(session_id)
        if not session:
            return [{"role": "system", "content": system_prompt}]

        return [{"role": "system", "content": system_prompt}] + session.history_messages

    def _apply_sliding_window(self, session: ReviewSession):
        """
        滑动窗口治理：当历史消息超过 2 * max_history_turns 条时截断最早的轮次，
        始终保证系统不超出本地 Qwen2.5 4096 Token 的承载限制
        """
        max_messages = self.max_history_turns * 2
        if len(session.history_messages) > max_messages:
            # 丢弃最早的多余轮次
            session.history_messages = session.history_messages[-max_messages:]

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


if __name__ == "__main__":
    print("[*] 正在测试 SessionManager...")
    mgr = SessionManager(max_history_turns=2)
    s_id = mgr.create_session()
    print(f"[+] 创建会话成功: {s_id}")

    # 模拟多轮对话并测试滑动窗口
    for i in range(6):
        mgr.append_message(s_id, "user", f"问题 {i}")
        mgr.append_message(s_id, "assistant", f"回答 {i}")

    session = mgr.get_session(s_id)
    print(f"[+] 滑动窗口后保留消息数: {len(session.history_messages)} (预期 4 条)")
    for msg in session.history_messages:
        print(f"  - {msg['role']}: {msg['content']}")