from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Belief:
    belief_id: str
    content: str
    belief_type: str
    status: str
    source: str
    turn_id: int
    parent_or_conflict_target: str | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BeliefLedger:
    def __init__(self) -> None:
        self._beliefs: list[Belief] = []
        self._events: list[dict[str, Any]] = []
        self._counter = 0

    def add_belief(
        self,
        *,
        content: str,
        belief_type: str,
        status: str,
        source: str,
        turn_id: int,
        confidence: float,
        parent_or_conflict_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Belief:
        self._counter += 1
        belief = Belief(
            belief_id=f"belief_{self._counter}",
            content=content,
            belief_type=belief_type,
            status=status,
            source=source,
            turn_id=turn_id,
            parent_or_conflict_target=parent_or_conflict_target,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._beliefs.append(belief)
        self._events.append(
            {
                "event": "add",
                "belief_id": belief.belief_id,
                "turn_id": turn_id,
                "status": status,
            }
        )
        return belief

    def get(self, belief_id: str) -> Belief:
        for belief in self._beliefs:
            if belief.belief_id == belief_id:
                return belief
        raise KeyError(f"Unknown belief: {belief_id}")

    def confirm_belief(self, belief_id: str, *, turn_id: int) -> Belief:
        belief = self.get(belief_id)
        belief.status = "confirmed"
        self._events.append({"event": "confirm", "belief_id": belief_id, "turn_id": turn_id})
        return belief

    def revise_belief(
        self,
        belief_id: str,
        *,
        new_content: str,
        turn_id: int,
        source: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> Belief:
        target = self.get(belief_id)
        target.status = "corrected"
        self._events.append({"event": "correct", "belief_id": belief_id, "turn_id": turn_id})
        return self.add_belief(
            content=new_content,
            belief_type=target.belief_type,
            status="confirmed",
            source=source,
            turn_id=turn_id,
            confidence=confidence,
            parent_or_conflict_target=belief_id,
            metadata=metadata,
        )

    def deprecate_belief(self, belief_id: str, *, turn_id: int) -> Belief:
        belief = self.get(belief_id)
        belief.status = "deprecated"
        self._events.append({"event": "deprecate", "belief_id": belief_id, "turn_id": turn_id})
        return belief

    def mark_unresolved(self, belief_id: str, *, turn_id: int) -> Belief:
        belief = self.get(belief_id)
        belief.status = "unresolved"
        self._events.append({"event": "unresolved", "belief_id": belief_id, "turn_id": turn_id})
        return belief

    def beliefs(self) -> list[Belief]:
        return list(self._beliefs)

    def snapshot(self) -> dict[str, Any]:
        return {
            "beliefs": [asdict(belief) for belief in self._beliefs],
            "events": list(self._events),
        }
