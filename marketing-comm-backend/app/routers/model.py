from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import traceback

from app.schemas.model import (
    LogisticModelStatusResponseSchema,
    BoostingModelStatusResponseSchema,
    TrainLogisticModelResponseSchema,
    TrainBoostingModelResponseSchema,
    PredictResponseSchema,
)
from app.services.ml_service import (
    train_logistic_model_from_file,
    train_boosting_model_from_file,
    get_logistic_model_status,
    get_boosting_model_status,
    predict_logistic_from_file,
    predict_boosting_from_file,
)
from app.services.file_service import get_prediction_file_path


router = APIRouter(prefix="/api/model", tags=["model"])

@router.get("/logistic/status", response_model=LogisticModelStatusResponseSchema)
def logistic_model_status():
    status = get_logistic_model_status()
    return {
        "success": True,
        "data": status,
    }


@router.post("/logistic/train", response_model=TrainLogisticModelResponseSchema)
def train_logistic_model(file: UploadFile = File(...)):
    try:
        result = train_logistic_model_from_file(file)
        return {
            "success": True,
            "data": result,
            "message": "Модель логистической регрессии успешно обучена.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.post("/logistic/predict", response_model=PredictResponseSchema)
def predict_logistic(file: UploadFile = File(...)):
    try:
        result = predict_logistic_from_file(file)
        return {
            "success": True,
            "data": result,
            "message": "Прогноз для модели логистической регрессии успешно рассчитан.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка при прогнозировании для модели логистической регрессии.",
        )


@router.get("/boosting/status", response_model=BoostingModelStatusResponseSchema)
def boosting_model_status():
    status = get_boosting_model_status()
    return {
        "success": True,
        "data": status,
    }


@router.post("/boosting/train", response_model=TrainBoostingModelResponseSchema)
def train_boosting_model(file: UploadFile = File(...)):
    try:
        result = train_boosting_model_from_file(file)
        return {
            "success": True,
            "data": result,
            "message": "Модель бустинга успешно обучена.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.post("/boosting/predict", response_model=PredictResponseSchema)
def predict_boosting(file: UploadFile = File(...)):
    try:
        result = predict_boosting_from_file(file)
        return {
            "success": True,
            "data": result,
            "message": "Прогноз для модели бустинга успешно рассчитан.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.get("/download-prediction/{prediction_id}")
def download_prediction(prediction_id: str):
    try:
        file_path = get_prediction_file_path(prediction_id)
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл результата не найден.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )