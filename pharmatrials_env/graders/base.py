from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractGrader(ABC):
    @abstractmethod
    def generate_ground_truth(
        self, documents: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def score(self, submission: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
