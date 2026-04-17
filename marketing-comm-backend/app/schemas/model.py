from typing import Optional

from pydantic import BaseModel


class MetricsSchema(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    rocAuc: float


class BaseModelStatusDataSchema(BaseModel):
    trained: bool
    algorithm: Optional[str] = None
    trainedAt: Optional[str] = None
    rowsCount: Optional[int] = None
    featuresCount: Optional[int] = None
    threshold: Optional[float] = None
    metrics: Optional[MetricsSchema] = None


class LogisticModelStatusDataSchema(BaseModelStatusDataSchema):
    classWeight: Optional[str] = None
    useScaler: Optional[bool] = None
    penalty: Optional[str] = None


class CatBoostModelStatusDataSchema(BaseModelStatusDataSchema):
    classWeights: Optional[list[int]] = None
    iterations: Optional[int] = None
    learningRate: Optional[float] = None
    depth: Optional[int] = None
    l2LeafReg: Optional[float] = None


class LogisticModelStatusResponseSchema(BaseModel):
    success: bool
    data: LogisticModelStatusDataSchema


class CatBoostModelStatusResponseSchema(BaseModel):
    success: bool
    data: CatBoostModelStatusDataSchema


class TrainLogisticModelResponseSchema(BaseModel):
    success: bool
    data: LogisticModelStatusDataSchema
    message: str


class TrainCatBoostModelResponseSchema(BaseModel):
    success: bool
    data: CatBoostModelStatusDataSchema
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