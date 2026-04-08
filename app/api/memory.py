
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time

Turn = Tuple[str, str]  # ("user"|"assistant", text)

@dataclass
class SessionMemory:
    turns: List[Turn] = field(default_factory=list)
    updated_at: float = field(default_factory=lambda: time.time())
    language: Optional[str] = None

class InMemorySessionStore:
    def __init__(self, max_turns: int = 8, ttl_minutes: int = 120):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_minutes * 60
        self._store: Dict[str, SessionMemory] = {}

    def _prune_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, mem in self._store.items() if now - mem.updated_at > self.ttl_seconds]
        for sid in expired:
            del self._store[sid]

    def get(self, session_id: str) -> SessionMemory:
        self._prune_expired()
        if session_id not in self._store:
            self._store[session_id] = SessionMemory()
        return self._store[session_id]

    def append(self, session_id: str, role: str, text: str) -> None:
        mem = self.get(session_id)
        mem.turns.append((role, text))
        mem.turns = mem.turns[-self.max_turns:]  # keep last N turns
        mem.updated_at = time.time()

    def set_language(self, session_id: str, language: str) -> None:
        mem = self.get(session_id)
        mem.language = language
        mem.updated_at = time.time()

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            del self._store[session_id]

def format_memory(turns: List[Turn]) -> str:
    """Compact transcript for prompt injection."""
    if not turns:
        return ""
    lines = ["Conversation so far (most recent last):"]
    for role, text in turns:
        prefix = "User" if role == "user" else "Assistant"
        # Keep it compact; avoid dumping huge text back in
        t = text.strip()
        if len(t) > 700:
            t = t[:700] + "…"
        lines.append(f"{prefix}: {t}")
    return "\n".join(lines)
