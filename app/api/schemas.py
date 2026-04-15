from enum import Enum
from pydantic import BaseModel, Field

class AnswerMode(str, Enum):
    professional = "professional"
    decision = "decision"
    playful = "playful"

class Verbosity(str, Enum):
    short = "short"
    normal = "normal"
    long = "long"

class CiteStyle(str, Enum):
    inline = "inline"
    end = "end"
    none = "none"




## add session id (for memory)
from typing import Optional
from pydantic import BaseModel, Field
# ... your enums stay the same ...

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    stand_id: Optional[int] = None   # ← ADD THIS
    mode: AnswerMode = AnswerMode.professional
    verbosity: Verbosity = Verbosity.normal
    language: str = "auto"
    cite_style: CiteStyle = CiteStyle.inline

    # memory controls
    session_id: Optional[str] = None
    use_session_memory: bool = True
