"""Connector-local statistics container.

Operation recording will be wired to the S04 trace design later; this type is
kept independent from both upstream Mooncake connector implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats


@dataclass
class MTSCConnectorStats(KVConnectorStats):
    def __post_init__(self) -> None:
        self.data = self.data or {"store": {}, "pd": {}}

    def reset(self) -> None:
        self.data = {"store": {}, "pd": {}}

    def is_empty(self) -> bool:
        return not any(self.data.get(component) for component in ("store", "pd"))

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not isinstance(other, MTSCConnectorStats):
            raise TypeError(f"Cannot aggregate {type(other)!r} into MTSC stats")
        for component in ("store", "pd"):
            target = self.data.setdefault(component, {})
            for name, values in other.data.get(component, {}).items():
                target.setdefault(name, []).extend(values)
        return self

    def reduce(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for component in ("store", "pd"):
            for name, values in self.data.get(component, {}).items():
                if values:
                    result[f"{component}_{name}_count"] = len(values)
                    result[f"{component}_{name}_sum"] = sum(values)
        return result
