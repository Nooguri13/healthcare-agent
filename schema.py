"""Pydantic schema for structured fields extracted from a discharge summary.

Validation is the point: the LLM proposes fields, Pydantic enforces types and
constraints, and anything malformed is rejected before it can reach the database.
Works with Pydantic v2 or v1.
"""
from __future__ import annotations

from typing import List, Optional

try:  # Pydantic v2
    from pydantic import BaseModel, Field, field_validator

    _V2 = True
except ImportError:  # pragma: no cover - v1 fallback
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore

    _V2 = False


class DischargeRecord(BaseModel):
    patient_id: str = Field(..., description="Facility patient identifier, e.g. PT-1001")
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, description="M or F as written in the document")
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    attending_physician: Optional[str] = None
    principal_diagnosis: Optional[str] = None
    allergies: Optional[str] = None
    medications: List[str] = Field(default_factory=list)

    if _V2:
        @field_validator("sex")
        @classmethod
        def _norm_sex(cls, v):
            if v is None:
                return v
            v = v.strip().upper()[:1]
            return v if v in {"M", "F"} else None
    else:  # pragma: no cover
        @field_validator("sex")
        def _norm_sex(cls, v):  # type: ignore
            if v is None:
                return v
            v = v.strip().upper()[:1]
            return v if v in {"M", "F"} else None

    def dict_compat(self) -> dict:
        return self.model_dump() if _V2 else self.dict()
