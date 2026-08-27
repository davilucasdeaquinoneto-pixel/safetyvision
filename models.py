from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Severity = Literal["high", "medium", "low"]
Confidence = Literal["high", "medium", "low"]


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def fit_inside_image(self) -> "BoundingBox":
        if self.x + self.width > 1.01 or self.y + self.height > 1.01:
            raise ValueError("A caixa precisa caber dentro da imagem.")
        return self


class RawRisk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=4, max_length=120)
    description: str = Field(min_length=8, max_length=500)
    category: str = Field(min_length=2, max_length=60)
    severity: Severity
    confidence: Confidence
    visual_evidence: str = Field(min_length=8, max_length=500)
    location_description: str = Field(default="", max_length=180)
    bounding_box: BoundingBox | None = None
    related_nrs: list[str] = Field(default_factory=list, max_length=5)
    recommendation: str = Field(min_length=8, max_length=600)
    corrective_actions: list[str] = Field(default_factory=list, max_length=6)

    @field_validator(
        "title",
        "description",
        "category",
        "visual_evidence",
        "location_description",
        "recommendation",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("corrective_actions", "related_nrs", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class RawVisionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    risks: list[RawRisk] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("limitations", mode="before")
    @classmethod
    def normalize_limitations(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class RiskItem(BaseModel):
    id: str
    title: str
    description: str
    category: str
    severity: Severity
    confidence: Literal["high", "medium"]
    visual_evidence: str
    location_description: str
    bounding_box: BoundingBox | None = None
    related_nrs: list[str] = Field(default_factory=list)
    recommendation: str
    corrective_actions: list[str] = Field(default_factory=list)
    source: Literal["visual"] = "visual"


class AnalysisSummary(BaseModel):
    risk_count: int = Field(ge=0)
    high_count: int = Field(ge=0)
    medium_count: int = Field(ge=0)
    low_count: int = Field(ge=0)
    highest_severity: Severity | None = None
    message: str


class ImageInfo(BaseModel):
    format: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AnalysisResponse(BaseModel):
    status: Literal["success"] = "success"
    analysis_id: int | None = None
    filename: str
    environment: str
    focuses: list[str]
    image: ImageInfo
    summary: AnalysisSummary
    limitations: list[str]
    risks: list[RiskItem]
    reported_observations: list[str]
    ai_analysis: dict[str, object] | None = None
