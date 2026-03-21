from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class ToolResult:
    status: str  # "success" | "warning" | "error"
    message: str
    data: dict = field(default_factory=dict)
    duration: float = 0.0


class BaseTool:
    name: str = "BaseTool"
    description: str = ""
    icon: str = ""

    def run(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            result = self._execute(**kwargs)
            result.duration = time.time() - start
            return result
        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Error en {self.name}: {str(e)}",
                duration=time.time() - start,
            )

    def _execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError
