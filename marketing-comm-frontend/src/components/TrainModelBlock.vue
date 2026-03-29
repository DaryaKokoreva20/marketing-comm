<script>
import { trainModel } from '../services/api'

export default {
  name: 'TrainModelBlock',
  emits: ['train-success'],
  data() {
    return {
      selectedFile: null,
      isLoading: false,
      errorMessage: '',
      successMessage: ''
    }
  },
  methods: {
    handleFileChange(event) {
      const file = event.target.files[0]
      this.selectedFile = file || null
      this.errorMessage = ''
      this.successMessage = ''
    },
    async submitTrain() {
      if (!this.selectedFile) {
        this.errorMessage = 'Сначала выберите файл.'
        return
      }

      this.isLoading = true
      this.errorMessage = ''
      this.successMessage = ''

      try {
        const response = await trainModel(this.selectedFile)

        this.successMessage = response.message || 'Модель успешно обучена.'
        this.$emit('train-success', response.data)
      } catch (error) {
        this.errorMessage = error.message || 'Не удалось выполнить обучение.'
      } finally {
        this.isLoading = false
      }
    }
  }
}
</script>

<template>
  <section class="card">
    <h2>Обучение модели</h2>

    <p class="description">
      Загрузите CSV или XLSX файл с историческими данными для обучения модели.
    </p>

    <input
      type="file"
      accept=".csv,.xlsx"
      @change="handleFileChange"
    />

    <button
      class="action-button"
      :disabled="!selectedFile || isLoading"
      @click="submitTrain"
    >
      {{ isLoading ? 'Обучение...' : 'Обучить модель на данных' }}
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
  </section>
</template>

<style scoped>
.card {
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 20px;
  background: #fff;
}

.description {
  color: #666;
  margin-bottom: 16px;
}

.action-button {
  display: block;
  margin-top: 16px;
  padding: 10px 16px;
  cursor: pointer;
}

.file-name {
  margin-top: 12px;
  color: #444;
}

.error {
  margin-top: 12px;
  color: #b00020;
}

.success {
  margin-top: 12px;
  color: #0a7a2f;
}
</style>