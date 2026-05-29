# Predicción de informalidad laboral en el Perú mediante Machine Learning

## Descripción del proyecto

Este proyecto tiene como objetivo predecir la probabilidad de informalidad laboral en el Perú mediante técnicas de *Machine Learning*, utilizando microdatos de la Encuesta Nacional de Hogares (ENAHO) 2023 elaborada por el Instituto Nacional de Estadística e Informática (INEI).

El estudio busca aproximar empíricamente qué características individuales están asociadas a una mayor probabilidad de empleo informal, incorporando variables socioeconómicas básicas como edad, sexo y nivel educativo.

Asimismo, el proyecto compara el desempeño de modelos econométricos interpretables y algoritmos de aprendizaje automático con el fin de evaluar su capacidad predictiva en el contexto del mercado laboral peruano.

---

## Objetivo

Desarrollar modelos de Machine Learning capaces de predecir la condición de informalidad laboral de trabajadores peruanos utilizando información individual proveniente de ENAHO 2023.

---

## Fuente de datos

**Encuesta Nacional de Hogares (ENAHO) 2023 – INEI**

Se emplearon los siguientes módulos:

* **Módulo 05:** Empleo e ingresos
* **Módulo 02:** Características de los miembros del hogar
* **Módulo 03:** Educación

---

## Variable objetivo

### Informalidad laboral

Se construyó una variable binaria:

* **1 = empleo informal**
* **0 = empleo formal**

A partir de la variable `OCUPINF` del módulo de empleo de ENAHO.

---

## Variables explicativas

Las variables utilizadas en esta primera versión del modelo fueron:

| Variable  | Descripción                   |
| --------- | ----------------------------- |
| Edad      | Edad del trabajador           |
| Mujer     | Variable dicotómica de sexo   |
| Educación | Años aproximados de educación |

---

## Metodología

El proyecto fue desarrollado siguiendo las siguientes etapas:

1. Limpieza y preprocesamiento de datos.
2. Integración de módulos ENAHO mediante identificadores del hogar y persona.
3. Construcción de variables explicativas.
4. División entrenamiento-prueba (*train-test split*).
5. Entrenamiento de modelos predictivos.
6. Evaluación del desempeño de los modelos.

---

## Modelos utilizados

### 1. Logistic Regression

Modelo estadístico interpretable utilizado para identificar asociaciones entre variables individuales e informalidad laboral.

### 2. Random Forest

Algoritmo de Machine Learning basado en árboles de decisión utilizado para capturar relaciones no lineales entre variables.

---

## Principales resultados

| Modelo              | Accuracy |
| ------------------- | -------: |
| Logistic Regression |   82.59% |
| Random Forest       |   83.00% |

Los resultados muestran que ambos modelos alcanzan un desempeño predictivo relativamente alto. No obstante, la regresión logística presentó un rendimiento cercano al de Random Forest, sugiriendo que gran parte de los patrones asociados a la informalidad laboral pueden ser capturados mediante relaciones relativamente lineales.

Asimismo, la educación emergió como el principal factor asociado a una menor probabilidad de informalidad laboral.

---

## Estructura del proyecto

```text
Proyecto_Informalidad_ML/
│
├── data_raw/
├── data_clean/
├── graficos/
├── informe/
├── outputs/
├── python/
│   ├── 01_preprocesamiento.py
│   └── 02_modelado.py
│
├── notebooks/
│   └── analisis_informalidad.ipynb
│
├── README.md
└── requirements.txt
```

---

## Tecnologías utilizadas

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Autor

**Gary Alca Chipana**
Estudiante de Economía – Universidad Nacional Mayor de San Marcos (UNMSM)

Proyecto desarrollado con fines de aprendizaje aplicado en ciencia de datos, Machine Learning y economía laboral.
