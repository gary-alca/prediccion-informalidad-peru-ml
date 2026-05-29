# ==========================================
# PROYECTO:
# Predicción de informalidad laboral
# en el Perú mediante Machine Learning
#
# Archivo:
# 02_modelado.py
#
# Objetivo:
# Entrenar y evaluar modelos de
# Machine Learning para predecir
# informalidad laboral usando ENAHO 2023
# ==========================================


# ==========================================
# 1. IMPORTAR LIBRERÍAS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 2. CARGAR DATASET LIMPIO
# ==========================================

ruta_dataset = (
    r"D:\Proyecto_Informalidad_ML"
    r"\data_clean\dataset_ml.csv"
)

datos_modelo = pd.read_csv(ruta_dataset)

print("Dataset cargado correctamente")
print(datos_modelo.shape)
print()


# ==========================================
# 3. DEFINIR VARIABLES
# ==========================================

X = datos_modelo[
    [
        "edad",
        "mujer",
        "educacion"
    ]
]

y = datos_modelo["informal"]

print("Variables definidas")
print()
print("Features:")
print(X.columns.tolist())
print()


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)

print("Train-test split realizado")
print()

print("Train:")
print(X_train.shape)

print()

print("Test:")
print(X_test.shape)
print()


# ==========================================
# 5. MODELO 1:
# LOGISTIC REGRESSION
# ==========================================

print("=" * 50)
print("MODELO 1: LOGISTIC REGRESSION")
print("=" * 50)

modelo_logit = LogisticRegression()

modelo_logit.fit(
    X_train,
    y_train
)

# Predicciones
y_pred_logit = modelo_logit.predict(
    X_test
)

# Accuracy
accuracy_logit = accuracy_score(
    y_test,
    y_pred_logit
)

print()
print("Accuracy Logistic Regression:")
print(round(accuracy_logit, 4))
print()

# Classification Report
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_logit
    )
)


# ==========================================
# 6. COEFICIENTES
# ==========================================

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente":
        modelo_logit.coef_[0]
})

coeficientes = (
    coeficientes
    .sort_values(
        by="Coeficiente"
    )
)

print()
print("Coeficientes del modelo:")
print(coeficientes)
print()


# ==========================================
# 7. MATRIZ DE CONFUSIÓN
# LOGISTIC REGRESSION
# ==========================================

cm_logit = confusion_matrix(
    y_test,
    y_pred_logit
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_logit
)

disp.plot()

plt.title(
    "Matriz de confusión - Logistic Regression"
)

plt.savefig(
    r"D:\Proyecto_Informalidad_ML"
    r"\graficos\matriz_logistic.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 8. GRÁFICO COEFICIENTES
# ==========================================

plt.figure(figsize=(8, 5))

plt.barh(
    coeficientes["Variable"],
    coeficientes["Coeficiente"]
)

plt.axvline(x=0)

plt.title(
    "Factores asociados a la informalidad laboral en el Perú"
)

plt.xlabel(
    "Coeficiente estimado"
)

plt.ylabel(
    "Variables"
)

plt.grid(True)

plt.savefig(
    r"D:\Proyecto_Informalidad_ML"
    r"\graficos\coeficientes_logistic.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 9. MODELO 2:
# RANDOM FOREST
# ==========================================

print("=" * 50)
print("MODELO 2: RANDOM FOREST")
print("=" * 50)

modelo_rf = (
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
)

modelo_rf.fit(
    X_train,
    y_train
)

# Predicciones
y_pred_rf = modelo_rf.predict(
    X_test
)

# Accuracy
accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

print()
print("Accuracy Random Forest:")
print(round(accuracy_rf, 4))
print()

# Classification Report
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# ==========================================
# 10. MATRIZ DE CONFUSIÓN
# RANDOM FOREST
# ==========================================

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf
)

disp.plot()

plt.title(
    "Matriz de confusión - Random Forest"
)

plt.savefig(
    r"D:\Proyecto_Informalidad_ML"
    r"\graficos\matriz_random_forest.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 11. COMPARACIÓN DE MODELOS
# ==========================================

resultados_modelos = pd.DataFrame({
    "Modelo": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_logit,
        accuracy_rf
    ]
})

print()
print("Comparación modelos:")
print(resultados_modelos)
print()


# ==========================================
# 12. EXPORTAR RESULTADOS
# ==========================================

ruta_output = (
    r"D:\Proyecto_Informalidad_ML"
    r"\outputs\comparacion_modelos.xlsx"
)

resultados_modelos.to_excel(
    ruta_output,
    index=False
)

print("Resultados exportados")
print(ruta_output)
print()


# ==========================================
# 13. GRÁFICO COMPARATIVO
# ==========================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    resultados_modelos["Modelo"],
    resultados_modelos["Accuracy"]
)

# Etiquetas de porcentaje
for bar in bars:

    altura = bar.get_height()

    plt.text(
        bar.get_x() + (
            bar.get_width() / 2
        ),
        altura + 0.002,
        f"{altura:.2%}",
        ha="center"
    )

plt.title(
    "Comparación de desempeño de modelos"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(0, 1)

plt.grid(True)

plt.savefig(
    r"D:\Proyecto_Informalidad_ML"
    r"\graficos\comparacion_modelos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 14. RESUMEN FINAL
# ==========================================

print("=" * 50)
print("RESUMEN FINAL")
print("=" * 50)

print()
print(
    "Logistic Regression:"
)

print(
    round(
        accuracy_logit,
        4
    )
)

print()

print(
    "Random Forest:"
)

print(
    round(
        accuracy_rf,
        4
    )
)

print()

if accuracy_rf > accuracy_logit:
    print(
        "Random Forest mostró mejor desempeño."
    )
else:
    print(
        "Logistic Regression mostró mejor desempeño."
    )

print()
print(
    "Proyecto ejecutado correctamente 😭🔥"
)
