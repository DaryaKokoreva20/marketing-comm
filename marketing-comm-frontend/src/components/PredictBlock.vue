<template>
  <section class="card">
    <h2>Прогнозирование</h2>

    <p class="description">
      Загрузите файл с новыми клиентами для расчета вероятности отклика с помощью модели {{ modelLabel }}.
    </p>

    <div v-if="!isModelReady" class="disabled-message">
      Прогнозирование недоступно, пока модель {{ modelLabel }} не обучена.
    </div>

    <div v-else>
      <input
        type="file"
        accept=".csv,.xlsx"
        @change="handleFileChange"
      />

      <button
        class="action-button"
        :disabled="!selectedFile || isLoading"
        @click="submitPredict"
      >
        {{ isLoading ? 'Расчет...' : `Спрогнозировать через ${modelLabel}` }}
      </button>

      <p v-if="selectedFile" class="file-name">
        Выбран файл: {{ selectedFile.name }}
      </p>

      <p v-if="errorMessage" class="error">
        {{ errorMessage }}
      </p>

      <p v-if="successMessage" class="success">
        {{ successMessage }}
      </p>

      <div v-if="predictionResult" class="result-box">
        <h3>Результат прогнозирования</h3>
        <p><strong>Имя файла:</strong> {{ predictionResult.fileName }}</p>
        <p><strong>Обработано клиентов:</strong> {{ predictionResult.rowsProcessed }}</p>
        <p><strong>Сгенерировано прогнозов:</strong> {{ predictionResult.predictionsGenerated }}</p>

        <button
          class="download-button"
          @click="handleDownload"
        >
          Скачать результат
        </button>
      </div>
    </div>
  </section>
</template>

<script>
import { predictForClients, downloadPredictionResult } from '../services/api'

export default {
  name: 'PredictBlock',
  props: {
    modelType: {
      type: String,
      required: true,
    },
    modelLabel: {
      type: String,
      required: true,
    },
    isModelReady: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['predict-success'],
  data() {
    return {
      selectedFile: null,
      isLoading: false,
      errorMessage: '',
      successMessage: '',
      predictionResult: null,
    }
  },
  methods: {
    handleFileChange(event) {
      this.selectedFile = event.target.files[0] || null
      this.errorMessage = ''
      this.successMessage = ''
      this.predictionResult = null
    },

    async submitPredict() {
      if (!this.selectedFile) {
        this.errorMessage = 'Сначала выберите файл.'
        return
      }

      this.isLoading = true
      this.errorMessage = ''
      this.successMessage = ''
      this.predictionResult = null

      try {
        const response = await predictForClients(this.modelType, this.selectedFile)
        this.successMessage = response.message || 'Прогноз успешно рассчитан.'
        this.predictionResult = response.data
        this.$emit('predict-success', response.data)
      } catch (error) {
        this.errorMessage = error.message || 'Не удалось выполнить прогнозирование.'
      } finally {
        this.isLoading = false
      }
    },

    async handleDownload() {
      try {
        await downloadPredictionResult(this.predictionResult)
      } catch (error) {
        this.errorMessage = error.message || 'Не удалось скачать результат.'
      }
    },
  },
}
</script>

<style scoped>
.card {
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 20px;
  background: #fff;
}

.description,
.disabled-message {
  color: #666;
}

.action-button,
.download-button {
  display: block;
  margin-top: 16px;
  padding: 10px 16px;
  cursor: pointer;
}

.file-name {
  margin-top: 12px;
}

.error {
  margin-top: 12px;
  color: #b00020;
}

.success {
  margin-top: 12px;
  color: #0a7a2f;
}

.result-box {
  margin-top: 20px;
  padding: 16px;
  border: 1px solid #e2e2e2;
  border-radius: 10px;
  background: #fafafa;
}
</style>