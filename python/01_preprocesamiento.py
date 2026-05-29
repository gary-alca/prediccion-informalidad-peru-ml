# ==========================================
# PROYECTO:
# Predicción de informalidad laboral
# en el Perú mediante Machine Learning
#
# Archivo:
# 01_preprocesamiento.py
#
# Objetivo:
# Construir un dataset limpio para
# Machine Learning a partir de ENAHO 2023
# ==========================================


# ==========================================
# 1. IMPORTAR LIBRERÍAS
# ==========================================

import pandas as pd


# ==========================================
# 2. DEFINIR RUTAS
# ==========================================

ruta_modulo_05 = (
    r"D:\Proyecto_Informalidad_ML"
    r"\data_raw\2023\906-Modulo05"
    r"\906-Modulo05\Enaho01a-2023-500.csv"
)

ruta_modulo_02 = (
    r"D:\Proyecto_Informalidad_ML"
    r"\data_raw\2023\906-Modulo02"
    r"\906-Modulo02\Enaho01-2023-200.csv"
)

ruta_modulo_03 = (
    r"D:\Proyecto_Informalidad_ML"
    r"\data_raw\2023\906-Modulo03"
    r"\906-Modulo03\Enaho01a-2023-300.csv"
)


# ==========================================
# 3. CARGAR BASES DE DATOS
# ==========================================

print("Cargando bases de datos...")

modulo_05 = pd.read_csv(
    ruta_modulo_05,
    encoding="latin1"
)

modulo_02 = pd.read_csv(
    ruta_modulo_02,
    encoding="latin1"
)

modulo_03 = pd.read_csv(
    ruta_modulo_03,
    encoding="latin1"
)

print("Bases cargadas correctamente")
print()


# ==========================================
# 4. CREAR VARIABLE OBJETIVO
# INFORMALIDAD LABORAL
# 1 = informal
# 0 = formal
# ==========================================

modulo_05["informal"] = (
    modulo_05["OCUPINF"]
    .map({
        "1": 1,
        "2": 0
    })
)

# Mantener solo observaciones válidas
datos_modelo = modulo_05[
    modulo_05["informal"].notna()
].copy()

print("Variable objetivo creada")
print(
    datos_modelo["informal"]
    .value_counts()
)
print()


# ==========================================
# 5. CREAR VARIABLES PERSONALES
# SEXO Y EDAD
# ==========================================

# Convertir edad a formato numérico
modulo_02["edad"] = pd.to_numeric(
    modulo_02["P208A"],
    errors="coerce"
)

# Crear variable sexo
# 0 = hombre
# 1 = mujer
modulo_02["mujer"] = (
    modulo_02["P207"]
    .map({
        "1": 0,
        "2": 1
    })
)

# Seleccionar variables necesarias
modulo_02_small = modulo_02[
    [
        "CONGLOME",
        "VIVIENDA",
        "HOGAR",
        "CODPERSO",
        "edad",
        "mujer"
    ]
]

print("Variables personales creadas")
print()


# ==========================================
# 6. CREAR VARIABLE EDUCACIÓN
# AÑOS APROXIMADOS DE EDUCACIÓN
# ==========================================

educacion_map = {
    1: 0,    # sin nivel
    2: 0,    # inicial
    3: 3,    # primaria incompleta
    4: 6,    # primaria completa
    5: 9,    # secundaria incompleta
    6: 11,   # secundaria completa
    7: 12,   # superior no universitaria incompleta
    8: 14,   # superior no universitaria completa
    9: 14,   # universitaria incompleta
    10: 16,  # universitaria completa
    11: 18,  # posgrado
    12: 0    # básica especial
}

modulo_03["educacion"] = (
    modulo_03["P301A"]
    .map(educacion_map)
)

# Seleccionar variables necesarias
modulo_03_small = modulo_03[
    [
        "CONGLOME",
        "VIVIENDA",
        "HOGAR",
        "CODPERSO",
        "educacion"
    ]
]

print("Variable educación creada")
print()


# ==========================================
# 7. MERGE DE VARIABLES PERSONALES
# ==========================================

datos_modelo = datos_modelo.merge(
    modulo_02_small,
    on=[
        "CONGLOME",
        "VIVIENDA",
        "HOGAR",
        "CODPERSO"
    ],
    how="left"
)

print("Merge módulo 02 completado")
print()


# ==========================================
# 8. MERGE DE EDUCACIÓN
# ==========================================

datos_modelo = datos_modelo.merge(
    modulo_03_small,
    on=[
        "CONGLOME",
        "VIVIENDA",
        "HOGAR",
        "CODPERSO"
    ],
    how="left"
)

print("Merge módulo 03 completado")
print()


# ==========================================
# 9. CONSTRUIR DATASET FINAL
# ==========================================

datos_modelo = datos_modelo[
    [
        "informal",
        "edad",
        "mujer",
        "educacion"
    ]
].dropna()

print("Dataset final creado")
print(datos_modelo.shape)
print()


# ==========================================
# 10. EXPORTAR DATASET LIMPIO
# ==========================================

ruta_output = (
    r"D:\Proyecto_Informalidad_ML"
    r"\data_clean\dataset_ml.csv"
)

datos_modelo.to_csv(
    ruta_output,
    index=False
)

print("Dataset exportado correctamente")
print(ruta_output)


# ==========================================
# 11. RESUMEN FINAL
# ==========================================

print()
print("Resumen del dataset final")
print(datos_modelo.head())

print()
print("Valores perdidos:")
print(datos_modelo.isna().sum())
