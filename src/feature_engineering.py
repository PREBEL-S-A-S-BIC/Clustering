# -*- coding: utf-8 -*-
"""
feature_engineering.py - Construccion de las variables del modelo.

A partir de los 30 meses de unidades vendidas se construyen tres bloques:

    A) Variables semestrales
       unidades_SX_YYYY        suma de unidades del semestre
       meses_con_venta_SX_YYYY meses con venta > 0 dentro del semestre

    B) Variables de comportamiento (las que alimentan el clustering)
       total_unidades      suma de los 30 meses
       meses_activos       meses con venta > 0
       promedio_mensual    total_unidades / meses_activos
       cobertura_meses     % de los 30 meses con venta
       tendencia           promedio mensual S2_2025 - promedio mensual S1_2024
       estacionalidad      ratio S1 vs S2 usando los dos anios completos
       variabilidad        coeficiente de variacion de los meses activos
       ingreso_estimado    total_unidades * PVP
       mejor_semestre      semestre con mayor venta historica
       canal_cod           canal codificado numericamente

    C) Atributos de producto codificados con One-Hot para el modelo.

Decisiones de calculo documentadas:
    - `promedio_mensual` se divide por meses ACTIVOS (no por 30) para medir la
      intensidad de venta cuando el producto si estuvo disponible. Un lanzamiento
      reciente no queda castigado por los meses en que aun no existia.
    - `estacionalidad` usa solo 2024 y 2025, los unicos anios con ambos
      semestres completos. 2026 solo tiene S1 y sesgaria el ratio.
    - `tendencia` compara el primer semestre observado (S1_2024) contra el
      ultimo semestre CERRADO comparable (S2_2025), en unidades por mes.
    - Los productos sin ninguna venta se marcan con `sin_movimiento = True` y
      quedan fuera del clustering principal, con cluster reservado -1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import ingesta, utils


# ---------------------------------------------------------------------------
# A) VARIABLES SEMESTRALES
# ---------------------------------------------------------------------------


def agregar_variables_semestrales(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega unidades y meses con venta por cada semestre del historico."""
    utils.log("Construyendo variables semestrales ...")
    df = df.copy()

    for semestre, meses in cfg.SEMESTRES.items():
        presentes = [m for m in meses if m in df.columns]
        if not presentes:
            utils.log(f"  {semestre}: sin meses disponibles, se omite", nivel="ALERTA")
            continue

        # Unidades vendidas en el semestre
        df[f"unidades_{semestre}"] = df[presentes].sum(axis=1).astype("int64")
        # Cuantos de esos meses tuvieron al menos una venta (mide continuidad)
        df[f"meses_con_venta_{semestre}"] = (df[presentes] > 0).sum(axis=1).astype("int16")
        # Promedio por mes del semestre: base para tendencia y proyeccion
        df[f"promedio_{semestre}"] = df[f"unidades_{semestre}"] / len(presentes)

        utils.log(f"  {semestre}: {len(presentes)} meses, "
                  f"{utils.formatear_miles(df[f'unidades_{semestre}'].sum())} unidades")
    return df


# ---------------------------------------------------------------------------
# B) VARIABLES DE COMPORTAMIENTO
# ---------------------------------------------------------------------------


def calcular_estacionalidad(df: pd.DataFrame) -> pd.Series:
    """Ratio de venta S1 vs S2 sobre los dos anios completos (2024 y 2025).

    Interpretacion:
        > 1  -> el producto vende mas en el primer semestre
        = 1  -> comportamiento equilibrado
        < 1  -> vende mas en el segundo semestre

    Casos borde: si el producto no vendio en ningun S2 pero si en S1, el ratio
    seria infinito; se acota a 3.0 (equivalente a "fuertemente S1") para no
    generar valores extremos que distorsionen el escalado del modelo. Si no
    vendio en ninguno de los dos, se asigna 1.0 (neutro).
    """
    cols_s1 = [f"unidades_{s}" for s in cfg.SEMESTRES_S1_COMPLETOS if f"unidades_{s}" in df.columns]
    cols_s2 = [f"unidades_{s}" for s in cfg.SEMESTRES_S2_COMPLETOS if f"unidades_{s}" in df.columns]

    venta_s1 = df[cols_s1].sum(axis=1)
    venta_s2 = df[cols_s2].sum(axis=1)

    ratio = pd.Series(1.0, index=df.index)  # neutro por defecto

    ambos = (venta_s1 > 0) & (venta_s2 > 0)
    ratio.loc[ambos] = venta_s1[ambos] / venta_s2[ambos]

    solo_s1 = (venta_s1 > 0) & (venta_s2 == 0)
    ratio.loc[solo_s1] = 3.0

    solo_s2 = (venta_s1 == 0) & (venta_s2 > 0)
    ratio.loc[solo_s2] = 1.0 / 3.0

    # Se acota el rango para que un producto marginal no domine el escalado.
    return ratio.clip(lower=1 / 3, upper=3.0).round(4)


def calcular_variabilidad(df: pd.DataFrame, meses: list[str]) -> pd.Series:
    """Coeficiente de variacion (std/mean) sobre los meses con venta.

    Se calcula solo sobre meses activos: incluir los ceros de un producto nuevo
    inflaria artificialmente la variabilidad de todos los lanzamientos.
    Un CV bajo indica venta estable y predecible (facil de planear); un CV alto
    indica venta a picos (requiere mas colchon de inventario).
    """
    valores = df[meses].to_numpy(dtype="float64")
    activos = valores > 0

    resultado = np.zeros(len(df))
    for i in range(len(df)):
        serie = valores[i, activos[i]]
        if serie.size >= 2 and serie.mean() > 0:
            # ddof=1: desviacion muestral, coherente con una muestra de meses
            resultado[i] = serie.std(ddof=1) / serie.mean()
        # Con 0 o 1 mes activo la variabilidad no es estimable: queda en 0.

    return pd.Series(resultado, index=df.index).round(4)


def calcular_tendencia(df: pd.DataFrame) -> pd.Series:
    """Diferencia de venta promedio mensual entre S2_2025 y S1_2024.

    Positivo = el producto crece; negativo = cae. Se expresa en unidades por
    mes para que sea comparable entre semestres de distinta longitud.
    """
    # Los semestres no se nombran fijos: se toman el primero y el ultimo que
    # esten COMPLETOS en el calendario detectado del archivo. Asi, al agregar
    # meses nuevos la tendencia se recalcula contra el periodo correcto en vez
    # de quedarse comparando siempre 2024 contra 2025.
    completos = cfg.semestres_completos()
    if len(completos) < 2:
        utils.log("Se necesitan 2 semestres completos para medir tendencia: "
                  "queda en 0", nivel="ALERTA")
        return pd.Series(0.0, index=df.index)

    inicial, final = completos[0], completos[-1]
    prom_final = df.get(f"promedio_{final}", pd.Series(0.0, index=df.index))
    prom_inicial = df.get(f"promedio_{inicial}", pd.Series(0.0, index=df.index))
    utils.log(f"  Tendencia: {final} contra {inicial}")
    return (prom_final - prom_inicial).round(4)


def determinar_mejor_semestre(df: pd.DataFrame) -> pd.Series:
    """Semestre con mayor venta historica de cada registro.

    Si el producto nunca vendio se devuelve 'Sin historial', que es informacion
    util para el comprador (producto nuevo o descontinuado de hecho).
    """
    cols = [f"unidades_{s}" for s in cfg.SEMESTRES if f"unidades_{s}" in df.columns]
    subset = df[cols]

    mejor = subset.idxmax(axis=1).str.replace("unidades_", "", regex=False)
    mejor[subset.max(axis=1) == 0] = "Sin historial"
    return mejor


def agregar_variables_comportamiento(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el bloque de variables que describe como se vende el producto."""
    utils.log("Construyendo variables de comportamiento ...")
    df = df.copy()
    meses = [m for m in cfg.MESES if m in df.columns]

    # --- Volumen -----------------------------------------------------------
    df["total_unidades"] = df[meses].sum(axis=1).astype("int64")
    df["meses_activos"] = (df[meses] > 0).sum(axis=1).astype("int16")

    # --- Intensidad: promedio sobre meses con venta ------------------------
    df["promedio_mensual"] = np.where(
        df["meses_activos"] > 0,
        df["total_unidades"] / df["meses_activos"].replace(0, np.nan),
        0.0,
    )
    df["promedio_mensual"] = df["promedio_mensual"].fillna(0.0).round(4)

    # --- Continuidad -------------------------------------------------------
    df["cobertura_meses"] = (100.0 * df["meses_activos"] / len(meses)).round(2)

    # --- Dinamica ----------------------------------------------------------
    df["tendencia"] = calcular_tendencia(df)
    df["estacionalidad"] = calcular_estacionalidad(df)
    df["variabilidad"] = calcular_variabilidad(df, meses)

    # --- Valor economico ---------------------------------------------------
    df["ingreso_estimado"] = (df["total_unidades"] * df["PVP"]).round(2)

    # --- Referencias descriptivas -----------------------------------------
    df["mejor_semestre"] = determinar_mejor_semestre(df)

    # Canal codificado numericamente (0,1,2) para reportes y filtros rapidos.
    # El modelo NO usa este codigo: usa el One-Hot, para no imponer un orden
    # artificial entre canales.
    canales = sorted(df["Canal"].unique())
    df["canal_cod"] = df["Canal"].map({c: i for i, c in enumerate(canales)}).astype("int16")

    # --- Marca de productos sin ninguna venta -----------------------------
    df["sin_movimiento"] = df["total_unidades"] == 0
    utils.log(
        f"  Registros sin movimiento: {int(df['sin_movimiento'].sum())} "
        f"({utils.porcentaje(int(df['sin_movimiento'].sum()), len(df))}%)"
    )

    return df


# ---------------------------------------------------------------------------
# C) CODIFICACION ONE-HOT DE LOS ATRIBUTOS
# ---------------------------------------------------------------------------


def codificar_categoricas(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
    min_frecuencia: int = 10,
) -> pd.DataFrame:
    """Aplica One-Hot Encoding a los atributos de producto.

    Dos cuidados practicos:
      1. Las columnas constantes (como 'Categoria' en esta fuente) generan una
         sola dummy sin varianza; se descartan porque no aportan al modelo.
      2. Los niveles con muy pocos registros se agrupan en 'Otros'. Sin esto,
         los 60 valores de 'Grupo' o los 50 de 'Tono' crearian decenas de
         columnas casi vacias que el clustering leeria como ruido.

    Devuelve solo las columnas dummy, con prefijo 'oh_'.
    """
    columnas = columnas or cfg.FEATURES_CATEGORICAS
    bloques = []

    for columna in columnas:
        if columna not in df.columns:
            utils.log(f"  '{columna}' no existe en el dataset, se omite", nivel="ALERTA")
            continue

        if df[columna].nunique() <= 1:
            utils.log(f"  '{columna}' es constante: descartada del modelo")
            continue

        serie = df[columna].astype(str)

        # Agrupacion de niveles poco frecuentes
        frecuencias = serie.value_counts()
        raros = frecuencias[frecuencias < min_frecuencia].index
        if len(raros):
            serie = serie.where(~serie.isin(raros), "Otros")
            utils.log(f"  '{columna}': {len(raros)} nivel(es) poco frecuentes -> 'Otros'")

        dummies = pd.get_dummies(serie, prefix=f"oh_{columna}", dtype="float64")
        bloques.append(dummies)
        utils.log(f"  '{columna}': {dummies.shape[1]} columnas One-Hot")

    if not bloques:
        return pd.DataFrame(index=df.index)

    return pd.concat(bloques, axis=1)


# ---------------------------------------------------------------------------
# PIPELINE DE FEATURES
# ---------------------------------------------------------------------------


def construir_features(maestro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ejecuta el paso 2 completo y devuelve el dataset de features."""
    utils.titulo("PASO 2 - FEATURE ENGINEERING")

    if maestro is None:
        maestro = ingesta.cargar_maestro()

    df = agregar_variables_semestrales(maestro)
    df = agregar_variables_comportamiento(df)

    # Las dummies se agregan al mismo DataFrame para que el clustering pueda
    # seleccionar sus columnas por prefijo sin recalcular nada.
    dummies = codificar_categoricas(df)
    df = pd.concat([df, dummies], axis=1)

    utils.log(f"Features construidas: {df.shape[1]} columnas totales "
              f"({dummies.shape[1]} de One-Hot)")

    resumen_features(df)
    return df


def resumen_features(df: pd.DataFrame) -> None:
    """Imprime estadisticas descriptivas de las variables de comportamiento."""
    utils.titulo("RESUMEN DE VARIABLES DE COMPORTAMIENTO")
    columnas = cfg.FEATURES_NUMERICAS + ["ingreso_estimado", "meses_activos"]
    existentes = [c for c in columnas if c in df.columns]
    print(df[existentes].describe().T.round(2).to_string())

    print("\nDistribucion de 'mejor_semestre':")
    print(df["mejor_semestre"].value_counts().to_string())


def guardar_features(df: pd.DataFrame) -> str:
    """Guarda las features en data/processed/features.parquet."""
    cfg.crear_directorios()
    df.to_parquet(cfg.ARCHIVO_FEATURES, index=False)
    utils.log(f"Features guardadas en {cfg.ARCHIVO_FEATURES}")
    return str(cfg.ARCHIVO_FEATURES)


def cargar_features() -> pd.DataFrame:
    """Lee las features ya procesadas; las construye si aun no existen."""
    if not cfg.ARCHIVO_FEATURES.exists():
        utils.log("features.parquet no existe todavia: ejecutando paso 2 ...")
        return ejecutar()
    features = pd.read_parquet(cfg.ARCHIVO_FEATURES)
    cfg.actualizar_calendario(ingesta.detectar_meses(features))
    return features


def ejecutar(maestro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Punto de entrada del paso 2."""
    features = construir_features(maestro)
    guardar_features(features)
    return features


if __name__ == "__main__":
    ejecutar()
