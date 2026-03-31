import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 60000,
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

export async function getModelStatus() {
  try {
    const response = await api.get('/api/model/status')
    return response.data
  } catch (error) {
    throw new Error(extractErrorMessage(error))
  }
}

export async function trainModel(file) {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/api/model/train', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  } catch (error) {
    throw new Error(extractErrorMessage(error))
  }
}

export async function predictForClients(file) {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/api/model/predict', formData, {
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

    const blob = new Blob([
      response.data,
    ], {
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
