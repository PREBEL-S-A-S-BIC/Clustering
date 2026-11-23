# -*- coding: utf-8 -*-
"""
clustering.py - Preprocesamiento, busqueda de k optimo y entrenamiento.

Flujo:
    1. Separar los productos 'Sin Movimiento' (cluster reservado -1).
    2. Escalar las variables numericas (StandardScaler) y unir el One-Hot.
    3. Reducir dimensionalidad con PCA conservando 90% de la varianza.
    4. Buscar el k optimo con Elbow + Silhouette + Davies-Bouldin.
    5. Entrenar KMeans, Aglomerativo Jerarquico y DBSCAN.
    6. Comparar con Silhouette / Calinski-Harabasz / Davies-Bouldin.
    7. Elegir el mejor modelo y asignar cluster a cada producto x canal.

Decisiones de modelado documentadas:
    - Las variables de volumen (total_unidades, promedio_mensual) tienen una
      distribucion muy sesgada (max 68.136 vs mediana 272). Se aplica log1p
      antes de escalar: sin esa transformacion los pocos productos estrella
      dominan la distancia euclidiana y el modelo termina separando
      "un top-seller" contra "todo lo demas".
    - El canal entra como One-Hot y no como codigo 0/1/2, para no imponer un
      orden artificial entre Exito, Farmatodo y Prosalon.
    - El eps de DBSCAN se calibra automaticamente recorriendo la curva de
      k-distancias hasta que el ruido baje del 10%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from . import feature_engineering as fe
from . import utils

# Variables de volumen que se transforman con log1p antes de escalar
VARIABLES_LOG = ["total_unidades", "promedio_mensual"]


# ---------------------------------------------------------------------------
# ESTRUCTURA DE RESULTADO
# ---------------------------------------------------------------------------


@dataclass
class ResultadoClustering:
    """Empaqueta todo lo que produce el paso 3, para usarlo en notebooks."""

    df: pd.DataFrame                  # dataset con la columna Cluster asignada
    matriz_escalada: np.ndarray       # datos escalados antes de PCA
    matriz_pca: np.ndarray            # datos proyectados por PCA
    columnas_modelo: list             # nombres de las columnas usadas
    scaler: StandardScaler
    pca: PCA
    k_optimo: int
    metricas_k: pd.DataFrame          # curva de metricas por k
    metricas_modelos: pd.DataFrame    # comparacion entre algoritmos
    modelo_ganador: str
    etiquetas: dict = field(default_factory=dict)  # etiquetas por algoritmo
    modelos: dict = field(default_factory=dict)    # objetos entrenados


# ---------------------------------------------------------------------------
# 1-3. PREPROCESAMIENTO
# ---------------------------------------------------------------------------


def separar_sin_movimiento(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en productos con y sin movimiento.

    Los que nunca vendieron no tienen comportamiento que agrupar: todas sus
    variables valen 0 y formarian un cluster degenerado que arruina las
    metricas. Se les asigna el cluster reservado -1.
    """
    con_movimiento = df.loc[~df["sin_movimiento"]].copy()
    sin_movimiento = df.loc[df["sin_movimiento"]].copy()

    utils.log(f"Registros para clustering: {len(con_movimiento):,}")
    utils.log(f"Registros 'Sin Movimiento' (excluidos, cluster "
              f"{cfg.CLUSTER_SIN_MOVIMIENTO}): {len(sin_movimiento):,}")
    return con_movimiento, sin_movimiento


def construir_matriz(df: pd.DataFrame) -> tuple[np.ndarray, list, StandardScaler]:
    """Escala las numericas y las une con las dummies One-Hot.

    Las dummies ya viven en 0/1, un rango comparable al de las numericas
    estandarizadas, asi que no se re-escalan: hacerlo amplificaria niveles
    poco frecuentes hasta dominar la distancia.
    """
    utils.log("Escalando variables del modelo ...")

    numericas = [c for c in cfg.FEATURES_NUMERICAS if c in df.columns]
    dummies = [c for c in df.columns if c.startswith("oh_")]

    bloque_num = df[numericas].astype("float64").copy()

    # log1p sobre las variables de volumen para comprimir la cola larga
    for columna in VARIABLES_LOG:
        if columna in bloque_num.columns:
            bloque_num[columna] = np.log1p(bloque_num[columna])
            utils.log(f"  log1p aplicado a '{columna}' (distribucion sesgada)")

    scaler = StandardScaler()
    escaladas = scaler.fit_transform(bloque_num)

    matriz = np.hstack([escaladas, df[dummies].to_numpy(dtype="float64")])
    columnas = numericas + dummies

    utils.log(f"  Matriz del modelo: {matriz.shape[0]} filas x {matriz.shape[1]} variables "
              f"({len(numericas)} numericas + {len(dummies)} One-Hot)")
    return matriz, columnas, scaler


def aplicar_pca(matriz: np.ndarray, varianza=None) -> tuple[np.ndarray, PCA]:
    """Reduce dimensionalidad conservando el % de varianza configurado."""
    varianza = varianza or cfg.VARIANZA_PCA
    pca = PCA(n_components=varianza, svd_solver="full", random_state=cfg.SEMILLA)
    proyeccion = pca.fit_transform(matriz)

    utils.log(
        f"PCA: {matriz.shape[1]} -> {proyeccion.shape[1]} componentes "
        f"({100 * pca.explained_variance_ratio_.sum():.2f}% de varianza conservada)"
    )
    return proyeccion, pca


# ---------------------------------------------------------------------------
# 4. NUMERO OPTIMO DE CLUSTERS
# ---------------------------------------------------------------------------


def evaluar_k(X: np.ndarray, k_min=None, k_max=None) -> pd.DataFrame:
    """Calcula inercia, Silhouette y Davies-Bouldin para un rango de k."""
    k_min = k_min or cfg.K_MIN
    k_max = k_max or cfg.K_MAX

    utils.log(f"Evaluando k de {k_min} a {k_max} ...")
    filas = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=20, random_state=cfg.SEMILLA)
        etiquetas = modelo.fit_predict(X)
        filas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(X, etiquetas),
            "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
            "davies_bouldin": davies_bouldin_score(X, etiquetas),
        })
        utils.log(f"  k={k:2d} | silhouette={filas[-1]['silhouette']:.4f} | "
                  f"davies_bouldin={filas[-1]['davies_bouldin']:.4f}")

    return pd.DataFrame(filas)


def k_por_codo(metricas: pd.DataFrame) -> int:
    """Detecta el codo de la curva de inercia por el metodo de la distancia.

    Se traza la recta entre el primer y el ultimo punto de la curva y se toma
    el k cuya distancia perpendicular a esa recta es maxima. Es el criterio
    grafico del codo, pero calculado y por tanto reproducible.
    """
    x = metricas["k"].to_numpy(dtype="float64")
    y = metricas["inercia"].to_numpy(dtype="float64")

    # Normalizacion a [0,1] para que las escalas de k e inercia sean comparables
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())

    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    vector = p2 - p1
    vector = vector / np.linalg.norm(vector)

    distancias = []
    for punto in np.column_stack([xn, yn]):
        proyeccion = p1 + np.dot(punto - p1, vector) * vector
        distancias.append(np.linalg.norm(punto - proyeccion))

    return int(x[int(np.argmax(distancias))])


def recomendar_k(metricas: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Recomienda el k optimo combinando los tres criterios.

    Cada criterio se normaliza a un puntaje 0-1 (donde 1 = mejor) y se
    promedia con pesos: Silhouette pesa mas porque mide directamente la
    separacion y cohesion de los grupos, que es lo que importa para un
    segmento de compra interpretable.
    """
    tabla = metricas.copy()

    def normalizar(serie, mayor_es_mejor=True):
        rango = serie.max() - serie.min()
        if rango == 0:
            return pd.Series(0.5, index=serie.index)
        escala = (serie - serie.min()) / rango
        return escala if mayor_es_mejor else 1 - escala

    tabla["p_silhouette"] = normalizar(tabla["silhouette"], True)
    tabla["p_davies_bouldin"] = normalizar(tabla["davies_bouldin"], False)
    tabla["p_calinski"] = normalizar(tabla["calinski_harabasz"], True)
    tabla["puntaje"] = (
        0.50 * tabla["p_silhouette"]
        + 0.30 * tabla["p_davies_bouldin"]
        + 0.20 * tabla["p_calinski"]
    ).round(4)

    k_codo = k_por_codo(metricas)
    k_silhouette = int(tabla.loc[tabla["silhouette"].idxmax(), "k"])
    k_davies = int(tabla.loc[tabla["davies_bouldin"].idxmin(), "k"])
    k_puntaje = int(tabla.loc[tabla["puntaje"].idxmax(), "k"])

    rango_silhouette = float(tabla["silhouette"].max() - tabla["silhouette"].min())

    utils.titulo("RECOMENDACION DE K")
    utils.log(f"Metodo del codo (inercia)      -> k = {k_codo}")
    utils.log(f"Mejor Silhouette (max)         -> k = {k_silhouette}")
    utils.log(f"Mejor Davies-Bouldin (min)     -> k = {k_davies}")
    utils.log(f"Puntaje combinado (ponderado)  -> k = {k_puntaje}")
    utils.log(f"Amplitud de la curva Silhouette: {rango_silhouette:.4f}")

    # Regla de decision:
    # Si la curva de Silhouette es practicamente plana (amplitud < 0.05), los
    # datos no tienen una separacion natural dominante: elegir el k de maximo
    # Silhouette seria quedarse con una diferencia dentro del ruido, y ese
    # maximo casi siempre cae en k=2 ("vende mucho / vende poco"), que no
    # permite diferenciar politicas de compra. En ese escenario manda el codo
    # de la inercia, que si identifica donde deja de ganarse compacidad.
    if rango_silhouette < 0.05:
        k_final = k_codo
        utils.log(
            "Curva Silhouette plana: se prioriza el metodo del codo, mas "
            "informativo cuando no hay una separacion natural dominante.",
            nivel="ALERTA",
        )
    else:
        k_final = k_puntaje

    # Piso de negocio: menos de 3 grupos no sirve para segmentar compras.
    if k_final < 3:
        utils.log(f"k = {k_final} elevado al minimo operativo de 3 grupos",
                  nivel="ALERTA")
        k_final = 3

    utils.log(f"K OPTIMO SELECCIONADO: {k_final}")
    return int(k_final), tabla


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO DE LOS TRES ALGORITMOS
# ---------------------------------------------------------------------------


def entrenar_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """KMeans con el k optimo. Es el modelo de referencia del proyecto."""
    utils.log(f"Entrenando KMeans (k={k}) ...")
    modelo = KMeans(n_clusters=k, n_init=50, random_state=cfg.SEMILLA)
    return modelo.fit_predict(X), modelo


def entrenar_jerarquico(X: np.ndarray, k: int, enlace="ward"):
    """Clustering aglomerativo jerarquico con enlace de Ward."""
    utils.log(f"Entrenando Aglomerativo Jerarquico (k={k}, enlace={enlace}) ...")
    modelo = AgglomerativeClustering(n_clusters=k, linkage=enlace)
    return modelo.fit_predict(X), modelo


def calcular_linkage(X: np.ndarray, muestra_maxima=1200, enlace="ward"):
    """Matriz de enlace para el dendrograma.

    Si hay muchas filas se toma una muestra aleatoria: el dendrograma completo
    es ilegible y el calculo es O(n^2) en memoria.
    """
    if len(X) > muestra_maxima:
        rng = np.random.default_rng(cfg.SEMILLA)
        indices = rng.choice(len(X), muestra_maxima, replace=False)
        utils.log(f"Dendrograma sobre una muestra de {muestra_maxima} registros")
        return linkage(X[indices], method=enlace), indices

    return linkage(X, method=enlace), np.arange(len(X))


def graficar_dendrograma(matriz_enlace, k: int, ax=None):
    """Dibuja el dendrograma con una linea de corte en k grupos."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    dendrogram(matriz_enlace, truncate_mode="lastp", p=30,
               no_labels=True, ax=ax, color_threshold=matriz_enlace[-(k - 1), 2])
    ax.axhline(matriz_enlace[-(k - 1), 2], color="red", linestyle="--",
               label=f"Corte en k = {k}")
    ax.set_title("Dendrograma - Clustering Jerarquico (enlace Ward)")
    ax.set_xlabel("Grupos de productos (nodos agrupados)")
    ax.set_ylabel("Distancia de Ward")
    ax.legend()
    return ax


def curva_k_distancias(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Distancias al k-esimo vecino, ordenadas. Base para calibrar eps."""
    vecinos = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distancias, _ = vecinos.kneighbors(X)
    return np.sort(distancias[:, -1])


def entrenar_dbscan(X: np.ndarray, max_ruido=None, min_samples=None):
    """DBSCAN con eps calibrado automaticamente.

    Estrategia: se recorren percentiles de la curva de k-distancias de menor a
    mayor. Un eps pequeno marca casi todo como ruido; a medida que crece, el
    ruido baja. Se toma el primer eps que deja el ruido por debajo del techo
    configurado (10%) y produce al menos 2 clusters.
    """
    max_ruido = max_ruido if max_ruido is not None else cfg.MAX_RUIDO_DBSCAN
    # Regla practica: min_samples ~ 2 * dimensiones, acotado para no exigir
    # densidades imposibles en un dataset de ~1.400 filas.
    min_samples = min_samples or int(np.clip(2 * X.shape[1], 5, 20))

    utils.log(f"Calibrando DBSCAN (min_samples={min_samples}, "
              f"ruido objetivo < {max_ruido:.0%}) ...")

    distancias = curva_k_distancias(X, min_samples)
    candidatos = np.percentile(distancias, np.arange(50, 100, 2))

    mejor = None
    for eps in candidatos:
        etiquetas = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X)
        ruido = float((etiquetas == -1).mean())
        n_clusters = len({e for e in etiquetas if e != -1})

        if n_clusters >= 2 and ruido < max_ruido:
            mejor = (float(eps), etiquetas, ruido, n_clusters)
            break
        # Se guarda el mejor intento por si ningun eps cumple el objetivo
        if mejor is None and n_clusters >= 2:
            mejor = (float(eps), etiquetas, ruido, n_clusters)

    if mejor is None:
        utils.log("DBSCAN no logro formar 2 o mas clusters con estos datos",
                  nivel="ALERTA")
        eps = float(np.percentile(distancias, 90))
        etiquetas = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        modelo = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return etiquetas, modelo

    eps, etiquetas, ruido, n_clusters = mejor
    utils.log(f"  eps = {eps:.4f} | clusters = {n_clusters} | ruido = {ruido:.2%}")
    if ruido >= max_ruido:
        utils.log(f"  No se alcanzo la meta de ruido < {max_ruido:.0%}: los datos "
                  "no tienen una estructura de densidad clara.", nivel="ALERTA")

    modelo = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return etiquetas, modelo


# ---------------------------------------------------------------------------
# 6. COMPARACION DE MODELOS
# ---------------------------------------------------------------------------


def evaluar_etiquetas(X: np.ndarray, etiquetas: np.ndarray, nombre: str) -> dict:
    """Calcula las tres metricas internas para un conjunto de etiquetas.

    En DBSCAN el ruido (-1) se excluye del calculo: no es un grupo, y dejarlo
    dentro penalizaria injustamente el modelo.
    """
    etiquetas = np.asarray(etiquetas)
    validos = etiquetas != -1
    X_validos, etq_validos = X[validos], etiquetas[validos]
    n_clusters = len(set(etq_validos))

    fila = {
        "modelo": nombre,
        "n_clusters": n_clusters,
        "n_registros_agrupados": int(validos.sum()),
        "pct_ruido": round(100 * float((~validos).mean()), 2),
    }

    if n_clusters < 2:
        # Las metricas no estan definidas con menos de 2 grupos
        fila.update({"silhouette": np.nan, "calinski_harabasz": np.nan,
                     "davies_bouldin": np.nan})
        return fila

    fila["silhouette"] = round(silhouette_score(X_validos, etq_validos), 4)
    fila["calinski_harabasz"] = round(calinski_harabasz_score(X_validos, etq_validos), 2)
    fila["davies_bouldin"] = round(davies_bouldin_score(X_validos, etq_validos), 4)
    return fila


def comparar_modelos(X: np.ndarray, etiquetas_por_modelo: dict) -> pd.DataFrame:
    """Tabla comparativa de los algoritmos entrenados."""
    utils.titulo("COMPARACION DE MODELOS")
    tabla = pd.DataFrame(
        [evaluar_etiquetas(X, etq, nombre) for nombre, etq in etiquetas_por_modelo.items()]
    )
    print(tabla.to_string(index=False))
    return tabla


def seleccionar_mejor_modelo(tabla: pd.DataFrame) -> str:
    """Elige el modelo ganador por Silhouette, penalizando el exceso de ruido.

    Se descartan los modelos que dejan mas del 10% de los productos sin
    clasificar: para planear compras se necesita una recomendacion para cada
    SKU, no un modelo elegante que deje huecos.
    """
    candidatos = tabla.dropna(subset=["silhouette"])
    elegibles = candidatos[candidatos["pct_ruido"] <= 100 * cfg.MAX_RUIDO_DBSCAN]

    if elegibles.empty:
        utils.log("Ningun modelo cumple el limite de ruido: se evalua sobre todos",
                  nivel="ALERTA")
        elegibles = candidatos

    ganador = elegibles.loc[elegibles["silhouette"].idxmax(), "modelo"]
    utils.log(f"MODELO GANADOR: {ganador}")
    return str(ganador)


# ---------------------------------------------------------------------------
# 7. PIPELINE COMPLETO
# ---------------------------------------------------------------------------


def ordenar_clusters_por_volumen(df: pd.DataFrame) -> pd.DataFrame:
    """Renumera los clusters de mayor a menor volumen promedio.

    KMeans asigna los ids en orden arbitrario segun la semilla. Renumerarlos
    hace que el cluster 0 sea siempre el de mayor rotacion, lo que vuelve el
    reporte legible y estable entre ejecuciones.
    """
    df = df.copy()
    reales = df.loc[df["Cluster"] != cfg.CLUSTER_SIN_MOVIMIENTO]
    if reales.empty:
        return df

    orden = (reales.groupby("Cluster")["total_unidades"].mean()
                   .sort_values(ascending=False).index.tolist())
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    mapa[cfg.CLUSTER_SIN_MOVIMIENTO] = cfg.CLUSTER_SIN_MOVIMIENTO

    df["Cluster"] = df["Cluster"].map(mapa).astype("int16")
    return df


def ejecutar(features: pd.DataFrame | None = None,
             guardar: bool = True) -> ResultadoClustering:
    """Punto de entrada del paso 3: entrena, compara y asigna clusters."""
    utils.titulo("PASO 3 - MODELO DE CLUSTERING")

    if features is None:
        features = fe.cargar_features()

    # --- 1. Separar sin movimiento ----------------------------------------
    con_mov, sin_mov = separar_sin_movimiento(features)

    # --- 2-3. Escalado y PCA ----------------------------------------------
    matriz, columnas, scaler = construir_matriz(con_mov)
    matriz_pca, pca = aplicar_pca(matriz)

    # --- 4. K optimo -------------------------------------------------------
    metricas_k = evaluar_k(matriz_pca)
    k_optimo, metricas_k = recomendar_k(metricas_k)

    # --- 5. Entrenamiento --------------------------------------------------
    utils.titulo("ENTRENAMIENTO DE MODELOS")
    etq_kmeans, modelo_kmeans = entrenar_kmeans(matriz_pca, k_optimo)
    etq_jerarquico, modelo_jerarquico = entrenar_jerarquico(matriz_pca, k_optimo)
    etq_dbscan, modelo_dbscan = entrenar_dbscan(matriz_pca)

    etiquetas = {
        "KMeans": etq_kmeans,
        "Jerarquico_Ward": etq_jerarquico,
        "DBSCAN": etq_dbscan,
    }
    modelos = {
        "KMeans": modelo_kmeans,
        "Jerarquico_Ward": modelo_jerarquico,
        "DBSCAN": modelo_dbscan,
    }

    # --- 6. Comparacion y seleccion ---------------------------------------
    metricas_modelos = comparar_modelos(matriz_pca, etiquetas)
    ganador = seleccionar_mejor_modelo(metricas_modelos)

    # --- 7. Asignacion final ----------------------------------------------
    con_mov = con_mov.copy()
    con_mov["Cluster"] = np.asarray(etiquetas[ganador]).astype("int16")

    # El ruido de DBSCAN (-1) chocaria con el cluster reservado de los
    # 'Sin Movimiento'. Si el ganador es DBSCAN, el ruido se reetiqueta como un
    # grupo propio: son productos atipicos que igual necesitan recomendacion.
    if ganador == "DBSCAN" and (con_mov["Cluster"] == -1).any():
        nuevo_id = int(con_mov.loc[con_mov["Cluster"] != -1, "Cluster"].max()) + 1
        n_atipicos = int((con_mov["Cluster"] == -1).sum())
        con_mov.loc[con_mov["Cluster"] == -1, "Cluster"] = nuevo_id
        utils.log(f"{n_atipicos} registros marcados como ruido por DBSCAN se "
                  f"reagrupan en el cluster {nuevo_id} (atipicos)")

    # Guardar las coordenadas PCA: las usan las graficas y el dashboard.
    con_mov["pca_1"] = matriz_pca[:, 0]
    con_mov["pca_2"] = matriz_pca[:, 1] if matriz_pca.shape[1] > 1 else 0.0

    sin_mov = sin_mov.copy()
    sin_mov["Cluster"] = cfg.CLUSTER_SIN_MOVIMIENTO
    sin_mov["pca_1"] = np.nan
    sin_mov["pca_2"] = np.nan

    df_final = pd.concat([con_mov, sin_mov], ignore_index=True)
    df_final = ordenar_clusters_por_volumen(df_final)

    utils.titulo("DISTRIBUCION FINAL DE CLUSTERS")
    conteo = df_final["Cluster"].value_counts().sort_index()
    for cluster, n in conteo.items():
        etiqueta = ("Sin Movimiento" if cluster == cfg.CLUSTER_SIN_MOVIMIENTO
                    else f"Cluster {cluster}")
        print(f"       {etiqueta:>16}: {n:>5} registros "
              f"({utils.porcentaje(n, len(df_final))}%)")

    resultado = ResultadoClustering(
        df=df_final,
        matriz_escalada=matriz,
        matriz_pca=matriz_pca,
        columnas_modelo=columnas,
        scaler=scaler,
        pca=pca,
        k_optimo=k_optimo,
        metricas_k=metricas_k,
        metricas_modelos=metricas_modelos,
        modelo_ganador=ganador,
        etiquetas=etiquetas,
        modelos=modelos,
    )

    if guardar:
        cfg.crear_directorios()
        metricas_modelos.to_csv(cfg.ARCHIVO_METRICAS, index=False)
        metricas_k.to_csv(cfg.DIR_OUTPUT_DATA / "metricas_k.csv", index=False)
        utils.log(f"Metricas guardadas en {cfg.DIR_OUTPUT_DATA}")

    return resultado


if __name__ == "__main__":
    ejecutar()
