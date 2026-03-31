from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.schemas.model import (
    ModelStatusResponseSchema,
    TrainModelResponseSchema,
    PredictResponseSchema
)
from app.services.model_service import (
    get_model_status,
    train_model_from_file,
    predict_from_file,
    get_prediction_file_path
)

import traceback

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponseSchema)
def model_status():
    status = get_model_status()
    return {
        "success": True,
        "data": status
    }


@router.post("/train", response_model=TrainModelResponseSchema)
def train_model(file: UploadFile = File(...)):
    try:
        result = train_model_from_file(file)
        return {
            "success": True,
            "data": result,
            "message": "Модель успешно обучена."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponseSchema)
def predict(file: UploadFile = File(...)):
    try:
        result = predict_from_file(file)
        return {
            "success": True,
            "data": result,
            "message": "Прогноз успешно рассчитан."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Внутренняя ошибка при прогнозировании.")


@router.get("/download-prediction/{prediction_id}")
def download_prediction(prediction_id: str):
    try:
        file_path = get_prediction_file_path(prediction_id)
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл результата не найден.")
    except Exception:
        raise HTTPException(status_code=500, detail="Ошибка при скачивании результата.")