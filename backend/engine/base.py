from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseEngine(ABC):
    @abstractmethod
    def generate_response(
        self,
        user_text: str,
        state_result: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> str:
        """
        Generate conversational response based on state, context, and history.
        """
        pass
