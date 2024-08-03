
# Módulo de Generación de Embeddings con DistilBERT

Este módulo proporciona funciones para preprocesar texto y generar embeddings utilizando el modelo DistilBERT.

## Instalación

Ejecutar las siguientes líneas para instalar las dependencias necesarias:

```bash
!pip install spacy transformers torch
!python -m spacy download es_core_news_sm
```

- **spacy**: Librería de procesamiento de lenguaje natural.
- **transformers**: Librería de Hugging Face para modelos de NLP.
- **torch**: Librería de PyTorch para manejar modelos de deep learning.

## **Cargar el Modelo de spaCy para Preprocesamiento**

Utilizamos el modelo **es_core_news_sm** de spaCy para preprocesar el texto en español. Este modelo incluye funcionalidades como lematización, eliminación de stopwords, y segmentación de oraciones.

### **Función de Preprocesamiento**

- **Lematización**: Reduce las palabras a su forma base.
- **Eliminación de stopwords**: Palabras comunes que no aportan mucho significado, como "y", "el", "de".
- **Eliminación de puntuación**.

La función `preprocess_text(text:str) -> str` se encarga de realizar estas tareas. Recibe un texto y lo procesa eliminando stopwords, puntuación, y lematizando las palabras.

## **Tokenizador y Modelo DistilBERT**

Utilizamos un tokenizador y un modelo DistilBERT proporcionados por Hugging Face. El tokenizador convierte el texto en tokens (unidades de significado), y el modelo DistilBERT genera los embeddings (representaciones vectoriales) para estos tokens.

**DistilBertTokenizer.from_pretrained('distilbert-base-uncased')**: Utilizamos un tokenizador preentrenado de DistilBERT que maneja texto en inglés. Este tokenizador convierte el texto en tokens que el modelo puede procesar.

**DistilBertModel.from_pretrained('distilbert-base-uncased')**: Utilizamos un modelo DistilBERT preentrenado. Este modelo ha sido entrenado en un gran corpus de texto y es eficiente para generar embeddings.

La función `get_embeddings(text:str, model:DistilBertModel, tokenizer:DistilBertTokenizer) -> torch.Tensor` recibe un texto, un modelo y un tokenizador y devuelve los embeddings.

## **Resumen de Pasos**

1. Procesar el texto (remover stopwords, puntuación, etc.).
2. Generar los tokens utilizando DistilBertTokenizer.
3. Obtener los embeddings utilizando DistilBertModel.

## **Evaluación del Modelo Sin Fine-Tuning**

Antes de ajustar el modelo, se evalúa utilizando un conjunto de datos de validación para establecer una línea base.

## **Entrenamiento y Fine-Tuning del Modelo DistilBERT**

En esta sección se detalla el proceso de fine-tuning del modelo DistilBERT utilizando un conjunto de datos de reseñas de películas en español. El objetivo es ajustar los pesos del modelo DistilBERT preentrenado y entrenar una capa de clasificación añadida para la tarea específica de clasificación de sentimientos.

## **Pasos del Proceso de Fine-Tuning**

1. **Carga y Preprocesamiento de Datos**:
   - Se cargó el dataset `IMDB_Dataset_SPANISH.csv`, que contiene reseñas de películas y su respectivo sentimiento (positivo o negativo).
   - Se dividió el dataset en conjuntos de entrenamiento y validación.
   - Se tokenizaron los textos utilizando el tokenizador de DistilBERT.

2. **Configuración del Modelo**:
   - Se utilizó el modelo `DistilBertForSequenceClassification` de Hugging Face, añadiendo una capa de clasificación al final.
   - La capa de clasificación es una capa densa que toma los embeddings generados por DistilBERT y produce una predicción de clase (positivo o negativo).

3. **Configuración del Entrenamiento**:
   - Se definieron los parámetros de entrenamiento, como el número de épocas, el tamaño del batch, los pasos de calentamiento (`warmup_steps`), y la tasa de decaimiento del peso (`weight_decay`).

4. **Entrenamiento del Modelo**:
   - Se entrenó el modelo utilizando el `Trainer` de Hugging Face, que maneja todo el ciclo de entrenamiento y evaluación.
   - Durante el entrenamiento, se ajustaron tanto los pesos del modelo DistilBERT como los de la capa de clasificación añadida.

5. **Evaluación y Guardado del Modelo**:
   - Se evaluó el rendimiento del modelo en el conjunto de datos de validación.
   - Se guardó el modelo entrenado junto con el tokenizador para su uso futuro.

## **Generación de Predicciones con el Modelo Ajustado**

Después de realizar el fine-tuning, se utiliza el modelo ajustado para realizar predicciones sobre ejemplos específicos.

## **Pipeline de Clasificación de Sentimientos**

Utilizamos un pipeline de Hugging Face para clasificar sentimientos en textos específicos.

1. **Cargar el Modelo y el Tokenizador**:
   - Se carga el modelo entrenado y el tokenizador desde la ruta `./sentiment_model`.

2. **Crear el Pipeline**:
   - Se crea un pipeline de clasificación de sentimientos utilizando `pipeline('sentiment-analysis')`.

3. **Realizar Predicciones**:
   - Se aplican las predicciones a textos específicos para determinar su sentimiento.

## **Documentación Adicional**

### **Justificación del Uso de DistilBERT**

DistilBERT fue seleccionado por su eficiencia y rendimiento en comparación con modelos más grandes como BERT. Su arquitectura más ligera lo hace ideal para tareas donde los recursos computacionales son limitados, sin sacrificar demasiado la precisión.

### **Preparación de Datos**

Se utilizó spaCy para el preprocesamiento, asegurando que los datos estuvieran en el formato adecuado para el modelo. El dataset fue dividido en entrenamiento y validación para evaluar el rendimiento del modelo de manera objetiva.

### **Integración del Modelo Encoder**

Se integró DistilBERT utilizando la librería `transformers` de Hugging Face, y se adaptó el pipeline del proyecto para utilizar este modelo en la tarea de clasificación de sentimientos.

### **Calidad del Código**

El código sigue las guías de estilo de PEP8, con docstrings claros y comentarios que facilitan su comprensión y mantenimiento. Además, se incluyeron instrucciones para la reproducción del entorno y la ejecución del proyecto en el archivo README.
