<script>
export default {
  name: 'ModelStatusBlock',
  props: {
    modelStatus: {
      type: Object,
      default: null
    },
    isStatusLoading: {
      type: Boolean,
      default: false
    },
    statusError: {
      type: String,
      default: ''
    }
  }
}
</script>

<template>
  <section class="card">
    <h2>Статус модели</h2>

    <p v-if="isStatusLoading" class="muted">
      Загрузка статуса модели...
    </p>

    <p v-else-if="statusError" class="error">
      {{ statusError }}
    </p>

    <div v-else-if="!modelStatus || !modelStatus.trained">
      <p class="muted">Модель еще не обучена.</p>
    </div>

    <div v-else>
      <p><strong>Статус:</strong> обучена</p>
      <p><strong>Алгоритм:</strong> {{ modelStatus.algorithm }}</p>
      <p><strong>Дата обучения:</strong> {{ modelStatus.trainedAt }}</p>
      <p><strong>Количество строк:</strong> {{ modelStatus.rowsCount }}</p>
      <p><strong>Количество признаков:</strong> {{ modelStatus.featuresCount }}</p>

      <div v-if="modelStatus.metrics" class="metrics">
        <h3>Метрики качества</h3>
        <p>Accuracy: {{ modelStatus.metrics.accuracy }}</p>
        <p>Precision: {{ modelStatus.metrics.precision }}</p>
        <p>Recall: {{ modelStatus.metrics.recall }}</p>
        <p>F1-score: {{ modelStatus.metrics.f1 }}</p>
        <p>ROC-AUC: {{ modelStatus.metrics.rocAuc }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.card {
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 20px;
  background: #fff;
}

.muted {
  color: #777;
}

.error {
  color: #b00020;
}

.metrics {
  margin-top: 16px;
}
</style>