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
      modelStatus: null,
      isStatusLoading: false,
      statusError: '',
      latestPredictionResult: null,
    }
  },
  computed: {
    isModelReady() {
      return Boolean(this.modelStatus && this.modelStatus.trained)
    },
  },
  methods: {
    async loadModelStatus() {
      this.isStatusLoading = true
      this.statusError = ''

      try {
        const response = await getModelStatus()
        this.modelStatus = response.data
      } catch (error) {
        this.statusError = error.message || 'Не удалось загрузить статус модели.'
      } finally {
        this.isStatusLoading = false
      }
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
</script>]

<template>
  <div class="page">
    <header class="page-header">
      <h1>Модуль прогнозирования отклика на маркетинговые коммуникации</h1>
      <p>
        Загрузка данных, обучение модели и прогнозирование вероятности покупки по новым клиентам.
      </p>
    </header>

    <main class="layout">
      <TrainModelBlock @train-success="handleTrainSuccess" />

      <ModelStatusBlock
        :modelStatus="modelStatus"
        :isStatusLoading="isStatusLoading"
        :statusError="statusError"
      />

      <PredictBlock
        :isModelReady="isModelReady"
        @predict-success="handlePredictSuccess"
      />
    </main>
  </div>
</template>

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
</style>
