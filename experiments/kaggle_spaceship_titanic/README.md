# Spaceship Titanic con TensorC

Clasificación binaria para la competencia [Spaceship Titanic de Kaggle](https://www.kaggle.com/competitions/spaceship-titanic). La red, el autograd y Adam usan TensorC; NumPy se usa para preparar los datos y calcular métricas.

## 1. Descargar datos

Inicia sesión en Kaggle, únete a la competencia y descarga sus datos. Coloca `train.csv` y `test.csv` en `experiments/kaggle_spaceship_titanic/data/`. Kaggle exige aceptar las reglas antes de habilitar la descarga.

## 2. Instalar y ejecutar

Desde la raíz del repositorio, usando Python 3.10 o posterior:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./python
python experiments/kaggle_spaceship_titanic/train.py
```

Para indicar otra carpeta o ajustar el número de épocas:

```sh
python experiments/kaggle_spaceship_titanic/train.py \
  --data-dir /ruta/a/los/csv --epochs 20 --batch-size 256
```

El programa separa una validación estratificada, codifica categorías one-hot, imputa y estandariza numéricos usando solo el conjunto de entrenamiento, y reporta exactitud frente a la clase mayoritaria. Después vuelve a entrenar con todos los datos etiquetados y predice el conjunto de Kaggle.

## 3. Archivos generados

Se guardan en `experiments/kaggle_spaceship_titanic/artifacts/`:

- `submission.csv`: columnas `PassengerId,Transported`, listo para subir a Kaggle.
- `spaceship_titanic.tck`: pesos del MLP TensorC.
- `spaceship_titanic.tck.json`: clases, arquitectura y parámetros de preprocesamiento.
- `metrics.json`: exactitud de validación, matriz de confusión y tiempos.

La validación local es una estimación; la puntuación oficial aparece después de subir `submission.csv` manualmente a Kaggle.

## Resultado en este Mac

Con la partición fija (semilla 42), semilla del modelo 7, 20 épocas y lotes de 256, una corrida local obtuvo **80.28 %** de exactitud de validación, frente a **79.87 %** de la corrida inicial. Kaggle dio un public score de **0.80289**. El entrenamiento final en CPU tomó menos de un segundo. La exactitud puede variar ligeramente entre corridas por el barajado de lotes; `metrics.json` guarda el resultado exacto de la última ejecución.
