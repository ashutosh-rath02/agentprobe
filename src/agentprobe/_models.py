from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class RecordedCall:
    request: Dict[str, Any]
    response: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    chunks: Optional[List[Dict[str, Any]]] = None  # populated for stream=True calls
