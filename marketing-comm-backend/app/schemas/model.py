from pydantic import BaseModel
from typing import Optional


class MetricsSchema(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    rocAuc: float


class ModelStatusDataSchema(BaseModel):
    trained: bool
    algorithm: Optional[str] = None
    trainedAt: Optional[str] = None
    rowsCount: Optional[int] = None
    featuresCount: Optional[int] = None
    threshold: Optional[float] = None
    metrics: Optional[MetricsSchema] = None


class ModelStatusResponseSchema(BaseModel):
    success: bool
    data: ModelStatusDataSchema


class TrainModelResponseSchema(BaseModel):
    success: bool
    data: ModelStatusDataSchema
    message: str


class PredictResultDataSchema(BaseModel):
    predictionId: str
    fileName: str
    rowsProcessed: int
    predictionsGenerated: int
    downloadUrl: str


class PredictResponseSchema(BaseModel):
    success: bool
    data: PredictResultDataSchema
    message: str