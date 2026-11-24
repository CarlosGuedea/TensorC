# Predicting Credit Card Fraud (Kaggle)

TensorC MLP para predecir la probabilidad `is_fraud` y optimizar ROC AUC. Kaggle exige un CSV con las columnas `id,is_fraud`.

## Ejecutar

Desde la raíz del repositorio, con la extensión TensorC compilada e instalada:

```sh
python experiments/kaggle_credit_card_fraud/train.py
```

El programa espera `train.csv`, `test.csv` y `sample_submission.csv` en `data/`. Separa una validación estratificada, normaliza variables numéricas, codifica variables categóricas, sobremuestrea fraudes en el entrenamiento y reporta ROC AUC. Después entrena el modelo final con todos los datos etiquetados.

Parámetros ajustables: `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--split-seed` y `--positive-fraction`.

## Resultados generados

`artifacts/` contiene `submission.csv`, el modelo `credit_fraud.tck` con metadatos de preprocesamiento y `metrics.json`. La validación local ROC AUC fue **0.93818**; la submission obtuvo **0.93915** de public score en Kaggle.

Los CSV de Kaggle y los artefactos locales están excluidos de Git mediante `.gitignore`.
