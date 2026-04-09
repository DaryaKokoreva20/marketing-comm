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
      <p><strong>Class weighting:</strong> {{ formatClassWeights(modelStatus) ?? '—'}}</p>
      <p><strong>Use scaler:</strong> {{ modelStatus.useScaler ?? '—' }}</p>
      <p><strong>Penalty:</strong> {{ modelStatus.penalty ?? '—' }}</p>

      <div v-if="modelStatus.metrics" class="metrics-box">
        <h3>Метрики</h3>
        <p><strong>Accuracy:</strong> {{ modelStatus.metrics.accuracy }}</p>
        <p><strong>Precision:</strong> {{ modelStatus.metrics.precision }}</p>
        <p><strong>Recall:</strong> {{ modelStatus.metrics.recall }}</p>
        <p><strong>F1:</strong> {{ modelStatus.metrics.f1 }}</p>
        <p><strong>ROC-AUC:</strong> {{ modelStatus.metrics.rocAuc }}</p>
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
  },
  methods: {
    formatClassWeights(modelStatus) {
      if (modelStatus.classWeights) {
        return Array.isArray(modelStatus.classWeights)
          ? modelStatus.classWeights.join(', ')
          : modelStatus.classWeights
      }

      if (modelStatus.classWeight) {
        return modelStatus.classWeight
      }

      return '—'
    },
  }
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