<template>
  <div class="page">
    <header class="page-header">
      <h1>Модуль прогнозирования отклика на маркетинговые коммуникации</h1>
      <p>
        Выберите модель, затем обучите ее на данных и выполните прогноз для новых клиентов.
      </p>
    </header>

    <section class="card model-selector-card">
      <h2>Выбор модели</h2>

      <label for="model-type" class="selector-label">
        Модель
      </label>

      <select
        id="model-type"
        v-model="selectedModelType"
        class="selector"
        @change="handleModelChange"
      >
        <option value="logistic">Logistic Regression</option>
        <option value="catboost">CatBoost</option>
      </select>
    </section>

    <main class="layout">
      <TrainModelBlock
        :modelType="selectedModelType"
        :modelLabel="selectedModelLabel"
        @train-success="handleTrainSuccess"
      />

      <ModelStatusBlock
        :modelStatus="modelStatus"
        :isStatusLoading="isStatusLoading"
        :statusError="statusError"
        :modelLabel="selectedModelLabel"
      />

      <PredictBlock
        :modelType="selectedModelType"
        :modelLabel="selectedModelLabel"
        :isModelReady="isModelReady"
        @predict-success="handlePredictSuccess"
      />
    </main>
  </div>
</template>

<script>
import TrainModelBlock from './components/TrainModelBlock.vue'
import ModelStatusBlock from './components/ModelStatusBlock.vue'
import PredictBlock from './components/PredictBlock.vue'
import { getModelStatus } from './services/api'

export default {
  name: 'App',
  components: {
    TrainModelBlock,
    ModelStatusBlock,
    PredictBlock,
  },
  data() {
    return {
      selectedModelType: 'logistic',
      modelStatus: null,
      isStatusLoading: false,
      statusError: '',
      latestPredictionResult: null,
    }
  },
  computed: {
    selectedModelLabel() {
      return this.selectedModelType === 'catboost'
        ? 'CatBoost'
        : 'Logistic Regression'
    },
    isModelReady() {
      return Boolean(this.modelStatus && this.modelStatus.trained)
    },
  },
  methods: {
    async loadModelStatus() {
      this.isStatusLoading = true
      this.statusError = ''

      try {
        const response = await getModelStatus(this.selectedModelType)
        this.modelStatus = response.data
      } catch (error) {
        this.modelStatus = null
        this.statusError = error.message || 'Не удалось загрузить статус модели.'
      } finally {
        this.isStatusLoading = false
      }
    },

    async handleModelChange() {
      this.latestPredictionResult = null
      await this.loadModelStatus()
    },

    handleTrainSuccess(payload) {
      this.modelStatus = payload
      this.statusError = ''
    },

    handlePredictSuccess(payload) {
      this.latestPredictionResult = payload
      console.log('Результат прогнозирования:', payload)
    },
  },
  mounted() {
    this.loadModelStatus()
  },
}
</script>

<style scoped>
.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px;
  font-family: Arial, sans-serif;
  background: #f7f8fa;
  min-height: 100vh;
}

.page-header {
  margin-bottom: 24px;
}

.page-header p {
  color: #666;
}

.layout {
  display: grid;
  gap: 20px;
}

.card {
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 20px;
  background: #fff;
}

.model-selector-card {
  margin-bottom: 20px;
}

.selector-label {
  display: block;
  margin-bottom: 8px;
  color: #444;
}

.selector {
  min-width: 260px;
  padding: 10px 12px;
}
</style>