import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 600000,
})

function extractErrorMessage(error) {
  if (error.response?.data?.detail) {
    return error.response.data.detail
  }

  if (error.response?.data?.message) {
    return error.response.data.message
  }

  if (error.message) {
    return error.message
  }

  return 'Произошла неизвестная ошибка.'
}

function getModelBasePath(modelType) {
  if (modelType === 'catboost') {
    return '/api/model/catboost'
  }

  return '/api/model/logistic'
}

export async function getModelStatus(modelType) {
  try {
    const basePath = getModelBasePath(modelType)
    const response = await api.get(`${basePath}/status`)
    return response.data
  } catch (error) {
    throw new Error(extractErrorMessage(error))
  }
}

export async function trainModel(modelType, file) {
  try {
    const basePath = getModelBasePath(modelType)
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post(`${basePath}/train`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  } catch (error) {
    throw new Error(extractErrorMessage(error))
  }
}

export async function predictForClients(modelType, file) {
  try {
    const basePath = getModelBasePath(modelType)
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post(`${basePath}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  } catch (error) {
    throw new Error(extractErrorMessage(error))
  }
}

export async function downloadPredictionResult(predictionResult) {
  if (!predictionResult || !predictionResult.predictionId || !predictionResult.fileName) {
    throw new Error('Нет данных для скачивания результата.')
  }

  try {
    const response = await api.get(
      `/api/model/download-prediction/${predictionResult.predictionId}`,
      {
        responseType: 'blob',
      }
    )

    const blob = new Blob([response.data], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    })

    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = predictionResult.fileName

    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    window.URL.revokeObjectURL(url)
  } catch (error) {
    throw new Error(extractErrorMessage(error))
  }
}