export async function getModelStatus() {
  return {
    success: true,
    data: {
      trained: false,
      algorithm: null,
      trainedAt: null,
      rowsCount: null,
      featuresCount: null,
      metrics: null
    }
  }
}

export async function trainModel(file) {
  if (!file) {
    throw new Error('Файл для обучения не передан.')
  }

  return {
    success: true,
    data: {
      trained: true,
      algorithm: 'Logistic Regression',
      trainedAt: new Date().toLocaleString(),
      rowsCount: 2000,
      featuresCount: 21,
      metrics: {
        accuracy: 0.81,
        precision: 0.77,
        recall: 0.73,
        f1: 0.75,
        rocAuc: 0.84
      }
    },
    message: 'Модель успешно обучена.'
  }
}

export async function predictForClients(file) {
  if (!file) {
    throw new Error('Файл для прогнозирования не передан.')
  }

  return {
    success: true,
    data: {
      predictionId: 'prediction_001',
      fileName: 'prediction_results.xlsx',
      rowsProcessed: 150,
      predictionsGenerated: 4200,
      downloadUrl: '/api/model/download-prediction/prediction_001'
    },
    message: 'Прогноз успешно рассчитан.'
  }
}

export async function downloadPredictionResult(predictionResult) {
  if (!predictionResult || !predictionResult.fileName) {
    throw new Error('Нет данных для скачивания результата.')
  }

  const fileContent = [
    'Это заглушка файла результата прогнозирования.',
    `Файл: ${predictionResult.fileName}`,
    `predictionId: ${predictionResult.predictionId}`,
    `rowsProcessed: ${predictionResult.rowsProcessed}`,
    `predictionsGenerated: ${predictionResult.predictionsGenerated}`
  ].join('\n')

  const blob = new Blob([fileContent], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  })

  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')

  link.href = url
  link.download = predictionResult.fileName
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)

  window.URL.revokeObjectURL(url)
}