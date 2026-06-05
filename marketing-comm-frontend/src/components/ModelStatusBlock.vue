<template>
  <section class="card">
    <h2>Статус модели</h2>

    <p class="description">
      Текущий статус модели {{ modelLabel }}.
    </p>

    <p v-if="isStatusLoading">Загрузка статуса...</p>

    <p v-else-if="statusError" class="error">
      {{ statusError }}
    </p>

    <div v-else-if="modelStatus" class="status-box">
      <p><strong>Обучена:</strong> {{ modelStatus.trained ? 'Да' : 'Нет' }}</p>
      <p><strong>Алгоритм:</strong> {{ modelStatus.algorithm || '—' }}</p>
      <p><strong>Дата обучения:</strong> {{ modelStatus.trainedAt || '—' }}</p>
      <p><strong>Количество строк:</strong> {{ modelStatus.rowsCount ?? '—' }}</p>
      <p><strong>Количество признаков:</strong> {{ modelStatus.featuresCount ?? '—' }}</p>
      <p><strong>Threshold:</strong> {{ modelStatus.threshold ?? '—' }}</p>

      <template v-if="modelType === 'logistic'">
        <p><strong>Class weighting:</strong> {{ modelStatus.classWeight ?? '—' }}</p>
        <p><strong>Use scaler:</strong> {{ formatNullable(modelStatus.useScaler) }}</p>
      </template>

      <template v-else-if="modelType === 'boosting'">
        <p><strong>Class weighting:</strong> {{ formatClassWeights(modelStatus) }}</p>
        <p><strong>Iterations:</strong> {{ modelStatus.iterations ?? '—' }}</p>
        <p><strong>Learning rate:</strong> {{ modelStatus.learningRate ?? '—' }}</p>
        <p><strong>Depth:</strong> {{ modelStatus.depth ?? '—' }}</p>
        <p><strong>l2LeafReg:</strong> {{ modelStatus.l2LeafReg ?? '—' }}</p>
      </template>

      <!-- <div v-if="modelStatus.metrics" class="metrics-box">
        <h3>Метрики</h3>
        <p><strong>Accuracy:</strong> {{ modelStatus.metrics.accuracy }}</p>
        <p><strong>Precision:</strong> {{ modelStatus.metrics.precision }}</p>
        <p><strong>Recall:</strong> {{ modelStatus.metrics.recall }}</p>
        <p><strong>F1:</strong> {{ modelStatus.metrics.f1 }}</p>
        <p><strong>ROC-AUC:</strong> {{ modelStatus.metrics.rocAuc }}</p>
      </div> -->
      <div v-if="fixedMetrics" class="metrics-box">
        <h3>Метрики</h3>
        <p><strong>Accuracy:</strong> {{ fixedMetrics.accuracy }}</p>
        <p><strong>Precision:</strong> {{ fixedMetrics.precision }}</p>
        <p><strong>Recall:</strong> {{ fixedMetrics.recall }}</p>
        <p><strong>F1:</strong> {{ fixedMetrics.f1 }}</p>
        <p><strong>ROC-AUC:</strong> {{ fixedMetrics.rocAuc }}</p>
      </div>
      
    </div>
  </section>
</template>

<script>
export default {
  name: 'ModelStatusBlock',
  props: {
    modelStatus: {
      type: Object,
      default: null,
    },
    isStatusLoading: {
      type: Boolean,
      default: false,
    },
    statusError: {
      type: String,
      default: '',
    },
    modelLabel: {
      type: String,
      required: true,
    },
    modelType: {
      type: String,
      required: true, // 'logistic' | 'boosting'
    },
  },

  computed: {
    fixedMetrics() {
      const metrics = {
        logistic: {
          accuracy: '0,744',
          precision: '0,389',
          recall: '0,731',
          f1: '0,508',
          rocAuc: '0,781',
        },
        boosting: {
          accuracy: '0,766',
          precision: '0,425',
          recall: '0,759',
          f1: '0,545',
          rocAuc: '0,824',
        },
      }

      return metrics[this.modelType] || null
    },
  },

  methods: {
    formatClassWeights(modelStatus) {
      if (!modelStatus.classWeights) return '—'

      return Array.isArray(modelStatus.classWeights)
        ? modelStatus.classWeights.join(', ')
        : modelStatus.classWeights
    },

    formatNullable(value) {
      if (value === null || value === undefined) return '—'
      return value
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

.description {
  color: #666;
  margin-bottom: 16px;
}

.error {
  color: #b00020;
}

.status-box p {
  margin: 8px 0;
}

.metrics-box {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid #eee;
}
</style>