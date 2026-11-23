# -*- coding: utf-8 -*-
"""
evaluation.py - Interpretacion de los clusters y recomendacion de compra.

Que produce:
    1. Perfil de cada cluster: moda de los atributos y promedio de las metricas.
    2. Nombre automatico legible por cluster
       ("Alta Rotacion - Labios - Ambos Semestres").
    3. Etiqueta de prioridad de compra (Alta / Media / Baja).
    4. Proyeccion de unidades para S2_2026 por producto y canal.
    5. Visualizaciones: PCA 2D, t-SNE, barras de perfil, tabla resumen.

El nombre del cluster se arma con cuatro rasgos leidos de los datos, no
escritos a mano: nivel de rotacion, dinamica (crecimiento / declive / nuevo),
subcategoria dominante y foco de canal o estacionalidad. Asi el nombre sigue
siendo correcto si cambian los datos o el k optimo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import clustering as cl
from . import config as cfg
from . import utils

# ---------------------------------------------------------------------------
# UMBRALES DE INTERPRETACION
# ---------------------------------------------------------------------------
# Estacionalidad: cuanto debe desviarse el ratio S1/S2 de 1 para llamarlo
# estacional. +-15% evita etiquetar como estacional un desbalance trivial.
UMBRAL_ESTACIONAL_S1 = 1.15
UMBRAL_ESTACIONAL_S2 = 0.85

# Un cluster se considera concentrado en un canal si ese canal reune mas del
# 60% de sus registros; por debajo de eso el cluster es transversal.
UMBRAL_CONCENTRACION_CANAL = 0.60

# Un cluster se considera "producto nuevo" si su historia se concentra en los
# semestres recientes y su cobertura de meses es baja.
UMBRAL_COBERTURA_NUEVO = 25.0


# ---------------------------------------------------------------------------
# 1. PERFILAMIENTO
# ---------------------------------------------------------------------------


def perfilar_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de perfiles: una fila por cluster.

    Combina la moda de los atributos cualitativos (que producto es) con el
    promedio de las metricas de comportamiento (como se vende).
    """
    utils.log("Perfilando clusters ...")

    columnas_moda = ["Categoria", "Subcategoria", "Marca", "Canal", "Tipo",
                     "Tono", "Grupo", "Tipo Portafolio", "mejor_semestre"]
    columnas_moda = [c for c in columnas_moda if c in df.columns]

    cols_semestre = [c for c in utils.columnas_semestre("unidades") if c in df.columns]

    filas = []
    for cluster, grupo in df.groupby("Cluster"):
        fila = {
            "Cluster": int(cluster),
            "n_registros": len(grupo),
            "pct_registros": utils.porcentaje(len(grupo), len(df)),
            "n_productos_unicos": grupo["Material"].nunique(),
        }

        # --- Perfil cualitativo: que tipo de producto predomina -----------
        for columna in columnas_moda:
            fila[f"moda_{columna}"] = utils.moda_segura(grupo[columna])

        # --- Metricas de comportamiento -----------------------------------
        fila.update({
            "total_unidades_suma": int(grupo["total_unidades"].sum()),
            "total_unidades_prom": round(grupo["total_unidades"].mean(), 2),
            "total_unidades_mediana": round(grupo["total_unidades"].median(), 2),
            "promedio_mensual_prom": round(grupo["promedio_mensual"].mean(), 2),
            "cobertura_meses_prom": round(grupo["cobertura_meses"].mean(), 2),
            "tendencia_prom": round(grupo["tendencia"].mean(), 2),
            "estacionalidad_prom": round(grupo["estacionalidad"].mean(), 4),
            "variabilidad_prom": round(grupo["variabilidad"].mean(), 4),
            "ingreso_estimado_suma": round(grupo["ingreso_estimado"].sum(), 2),
            "ingreso_estimado_prom": round(grupo["ingreso_estimado"].mean(), 2),
            "pvp_prom": round(grupo["PVP"].mean(), 2),
            "pct_tendencia_positiva": utils.porcentaje(
                int((grupo["tendencia"] > 0).sum()), len(grupo)),
        })

        # --- Volumen por semestre: revela el patron temporal del cluster --
        for columna in cols_semestre:
            fila[f"{columna}_prom"] = round(grupo[columna].mean(), 2)

        # --- Composicion por canal ----------------------------------------
        reparto = grupo["Canal"].value_counts(normalize=True)
        for canal in cfg.HOJAS_CANAL:
            fila[f"pct_{canal}"] = round(100 * float(reparto.get(canal, 0.0)), 2)

        filas.append(fila)

    return pd.DataFrame(filas).sort_values("Cluster").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. NOMBRE AUTOMATICO DEL CLUSTER
# ---------------------------------------------------------------------------


def _nivel_rotacion(promedio_mensual: float, cortes: tuple[float, float]) -> str:
    """Clasifica la rotacion comparando contra los cortes entre clusters.

    Los cortes son relativos al propio dataset (terciles del promedio mensual
    de los clusters), no umbrales fijos: asi la etiqueta sigue siendo valida
    aunque cambie la escala del negocio.
    """
    bajo, alto = cortes
    if promedio_mensual >= alto:
        return "Alta Rotacion"
    if promedio_mensual >= bajo:
        return "Rotacion Media"
    return "Bajo Movimiento"


def _rasgo_dinamica(perfil: pd.Series) -> str | None:
    """Detecta si el cluster es nuevo, crece o cae. None si es estable."""
    # Producto nuevo: poca historia acumulada y su mejor semestre es reciente.
    # "Reciente" son los dos ultimos semestres del calendario detectado, no dos
    # nombres fijos: al agregar meses al Excel la definicion se corre sola.
    recientes = list(cfg.SEMESTRES)[-2:]
    if (perfil["cobertura_meses_prom"] < UMBRAL_COBERTURA_NUEVO
            and str(perfil.get("moda_mejor_semestre", "")) in recientes):
        return "Ingreso Reciente"

    if perfil["tendencia_prom"] < cfg.UMBRAL_TENDENCIA_NEGATIVA:
        return "En Declive"
    if perfil["tendencia_prom"] > 5:
        return "En Crecimiento"
    return None


def _rasgo_temporal_o_canal(perfil: pd.Series) -> str:
    """Devuelve el rasgo de cierre: estacionalidad o foco de canal.

    Se prioriza la estacionalidad cuando existe, porque es la informacion que
    define en cual de las dos compras semestrales pedir el producto. Si el
    cluster no es estacional, se informa si esta concentrado en un canal.
    """
    estacionalidad = perfil["estacionalidad_prom"]
    if estacionalidad >= UMBRAL_ESTACIONAL_S1:
        return "Estacional S1"
    if estacionalidad <= UMBRAL_ESTACIONAL_S2:
        return "Estacional S2"

    # Sin estacionalidad marcada: se mira la concentracion de canal
    porcentajes = {canal: perfil.get(f"pct_{canal}", 0.0) for canal in cfg.HOJAS_CANAL}
    canal_top, pct_top = max(porcentajes.items(), key=lambda x: x[1])
    if pct_top >= 100 * UMBRAL_CONCENTRACION_CANAL:
        return canal_top
    return "Ambos Semestres"


def nombrar_clusters(perfiles: pd.DataFrame) -> dict[int, str]:
    """Genera un nombre descriptivo para cada cluster a partir de su perfil.

    Formato: "<Rotacion>[ - <Dinamica>] - <Subcategoria> - <Estacionalidad|Canal>"
    Ejemplos reales que produce este modelo:
        "Alta Rotacion - Labios - Ambos Semestres"
        "Bajo Movimiento - Ingreso Reciente - Rostro - Estacional S2"
    """
    utils.log("Generando nombres automaticos de cluster ...")

    reales = perfiles[perfiles["Cluster"] != cfg.CLUSTER_SIN_MOVIMIENTO]

    # Cortes relativos de rotacion: terciles del promedio mensual entre clusters
    if len(reales) >= 3:
        cortes = (
            float(reales["promedio_mensual_prom"].quantile(1 / 3)),
            float(reales["promedio_mensual_prom"].quantile(2 / 3)),
        )
    elif len(reales):
        mediana = float(reales["promedio_mensual_prom"].median())
        cortes = (mediana, mediana)
    else:
        cortes = (0.0, 0.0)

    nombres: dict[int, str] = {}
    usados: dict[str, int] = {}

    for _, perfil in perfiles.iterrows():
        cluster = int(perfil["Cluster"])

        # Los productos sin ninguna venta tienen nombre fijo
        if cluster == cfg.CLUSTER_SIN_MOVIMIENTO:
            nombres[cluster] = "Producto Nuevo Sin Historial"
            continue

        partes = [_nivel_rotacion(perfil["promedio_mensual_prom"], cortes)]

        dinamica = _rasgo_dinamica(perfil)
        if dinamica:
            partes.append(dinamica)

        partes.append(str(perfil.get("moda_Subcategoria", "Maquillaje")))
        partes.append(_rasgo_temporal_o_canal(perfil))

        nombre = " - ".join(partes)

        # Dos clusters pueden compartir rasgos dominantes; se desambigua con el
        # id para que el nombre siga siendo una llave unica en los reportes.
        if nombre in usados:
            nombre = f"{nombre} (G{cluster})"
        usados[nombre] = cluster

        nombres[cluster] = nombre
        utils.log(f"  Cluster {cluster}: {nombre}")

    return nombres


def asignar_nombres(df: pd.DataFrame, perfiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega la columna Nombre_Cluster al dataset y a la tabla de perfiles."""
    nombres = nombrar_clusters(perfiles)
    df = df.copy()
    perfiles = perfiles.copy()
    df["Nombre_Cluster"] = df["Cluster"].map(nombres)
    perfiles["Nombre_Cluster"] = perfiles["Cluster"].map(nombres)
    return df, perfiles


# ---------------------------------------------------------------------------
# 3. PRIORIDAD DE COMPRA
# ---------------------------------------------------------------------------


def asignar_prioridad(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada producto x canal en prioridad Alta / Media / Baja.

    Reglas (en orden de evaluacion, la primera que aplica manda):
        Sin Movimiento : nunca vendio -> no se compra por historico
        Alta           : cobertura > 70% Y tendencia positiva
        Baja           : cobertura < 40% O caida fuerte de tendencia
        Media          : el resto (cobertura intermedia o tendencia estable)

    El orden importa: un producto con cobertura alta pero en caida fuerte no
    debe salir como Alta solo por estar siempre disponible.
    """
    utils.log("Asignando prioridad de compra ...")
    df = df.copy()

    sin_movimiento = df["total_unidades"] == 0
    alta = (df["cobertura_meses"] > cfg.UMBRAL_COBERTURA_ALTA) & (df["tendencia"] > 0)
    baja = (
        (df["cobertura_meses"] < cfg.UMBRAL_COBERTURA_BAJA)
        | (df["tendencia"] < cfg.UMBRAL_TENDENCIA_NEGATIVA)
    )

    df["Prioridad_Compra"] = "Media"
    df.loc[baja, "Prioridad_Compra"] = "Baja"
    df.loc[alta, "Prioridad_Compra"] = "Alta"       # Alta pisa a Baja
    df.loc[sin_movimiento, "Prioridad_Compra"] = "Sin Movimiento"

    reparto = df["Prioridad_Compra"].value_counts()
    for etiqueta, n in reparto.items():
        utils.log(f"  {etiqueta}: {n} registros ({utils.porcentaje(n, len(df))}%)")

    return df


# ---------------------------------------------------------------------------
# 4. PROYECCION PARA S2_2026
# ---------------------------------------------------------------------------


def calcular_factor_cluster(df: pd.DataFrame) -> pd.Series:
    """Factor de paso S1 -> S2 por cluster, calibrado con 2025.

    Para cada cluster: cuantas unidades vendio en S2_2025 por cada unidad
    vendida en S1_2025. Es el patron semestral tipico del grupo y es lo que
    se aplica a S1_2026 para estimar S2_2026.

    Se calcula sobre el agregado del cluster (no producto a producto) porque
    ahi esta el valor del clustering: un SKU individual puede tener un semestre
    raro, el grupo promedia ese ruido.
    """
    # El par de semestres no va fijo: se toma el ultimo anio con los dos
    # semestres cerrados en el calendario detectado del archivo.
    sem_origen, sem_destino = cfg.par_semestral_reciente()
    if not sem_origen or not sem_destino:
        utils.log("No hay un anio con ambos semestres completos: el factor "
                  "queda en 1.0 (sin ajuste semestral)", nivel="ALERTA")
        return pd.Series(1.0, index=sorted(df["Cluster"].unique()))

    # Si lo que se planea es un primer semestre, el paso que interesa es el
    # inverso (de S2 al S1 siguiente), asi que se invierte el par.
    if cfg.SEMESTRE_OBJETIVO.startswith("S1"):
        sem_origen, sem_destino = sem_destino, sem_origen

    col_origen, col_destino = f"unidades_{sem_origen}", f"unidades_{sem_destino}"
    utils.log(f"  Calibrado con {sem_origen} -> {sem_destino}")

    agregado = df.groupby("Cluster")[[col_origen, col_destino]].sum()

    factor = np.where(
        agregado[col_origen] > 0,
        agregado[col_destino] / agregado[col_origen].replace(0, np.nan),
        1.0,
    )
    # Se acota entre 0,5x y 2x: un factor mas extremo casi siempre viene de un
    # cluster con base pequena y llevaria a sobre o sub-comprar de forma grave.
    factor = np.clip(np.nan_to_num(factor, nan=1.0), 0.5, 2.0)

    serie = pd.Series(factor, index=agregado.index).round(4)

    utils.log("Factor S1->S2 por cluster (calibrado con 2025):")
    for cluster, valor in serie.items():
        etiqueta = ("Sin Movimiento" if cluster == cfg.CLUSTER_SIN_MOVIMIENTO
                    else f"Cluster {cluster}")
        utils.log(f"  {etiqueta}: x{valor:.3f}")

    return serie


def proyectar_semestre_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    """Estima las unidades a pedir para S2_2026 por producto y canal.

    Jerarquia de metodos (se usa el primero que aplique):

    1. 'S1_2026 x factor cluster'
       El producto vendio en el semestre mas reciente. Se toma esa venta real y
       se ajusta por el patron S1->S2 de su cluster. Es el metodo mas confiable
       porque parte de un dato observado y actual.

    2. 'Promedio S2 propio'
       No vendio en S1_2026 pero si tiene historia en S2 anteriores. Se usa el
       promedio de sus propios segundos semestres: mismo semestre, mismo SKU.

    3. 'Promedio S2 del cluster'
       Producto con venta pero sin ningun S2 propio (tipicamente un lanzamiento
       reciente). Se usa el promedio de S2 de los productos activos de su
       cluster: es exactamente el caso que justifica el modelo.

    4. 'Sin historial - ver hoja Sugerido_Lanzamientos'
       Nunca vendio nada, asi que el historico no da base para proyectar y aqui
       queda en 0. La cifra para estos productos la calcula el modelo de
       atributos (`lanzamientos.py`), que estima desde canal, marca, grupo, tipo,
       tono, PVP y mes de lanzamiento. El 0 de esta hoja NO significa "no pedir":
       significa "la cantidad esta en la hoja Sugerido_Lanzamientos".
    """
    sem_objetivo = cfg.SEMESTRE_OBJETIVO
    utils.log(f"Proyectando {sem_objetivo} ...")
    df = df.copy()

    factores = calcular_factor_cluster(df)
    df["factor_cluster_S2"] = df["Cluster"].map(factores).fillna(1.0)

    # --- Base propia: promedio de los S2 en que el producto estuvo activo ---
    cols_s2 = [f"unidades_{s}" for s in cfg.SEMESTRES_HOMOLOGOS if f"unidades_{s}" in df.columns]
    valores_s2 = df[cols_s2]
    semestres_activos = (valores_s2 > 0).sum(axis=1)
    df["base_propia_S2"] = np.where(
        semestres_activos > 0,
        valores_s2.sum(axis=1) / semestres_activos.replace(0, np.nan),
        0.0,
    )
    df["base_propia_S2"] = df["base_propia_S2"].fillna(0.0).round(2)

    # --- Base del cluster: promedio de S2 entre los activos del cluster -----
    activos = df.loc[df["base_propia_S2"] > 0]
    base_cluster = activos.groupby("Cluster")["base_propia_S2"].mean()
    df["base_cluster_S2"] = df["Cluster"].map(base_cluster).fillna(0.0).round(2)

    # --- Aplicacion de la jerarquia de metodos -----------------------------
    # El semestre base es el ultimo con datos del calendario detectado, no un
    # nombre fijo: si se agregan meses al Excel, la proyeccion arranca del dato
    # mas reciente en vez de seguir anclada a S1_2026.
    sem_base = cfg.semestre_base()
    col_base = f"unidades_{sem_base}"
    utils.log(f"  Semestre base (dato mas reciente): {sem_base}")

    tiene_s1_2026 = df.get(col_base, pd.Series(0, index=df.index)) > 0
    tiene_s2_propio = df["base_propia_S2"] > 0
    sin_movimiento = df["total_unidades"] == 0

    proyeccion = pd.Series(0.0, index=df.index)
    metodo = pd.Series("Sin historial - ver hoja Sugerido_Lanzamientos", index=df.index)

    # Metodo 3 (base): productos con venta pero sin S2 propio
    caso3 = ~sin_movimiento & ~tiene_s2_propio & ~tiene_s1_2026
    proyeccion.loc[caso3] = df.loc[caso3, "base_cluster_S2"]
    metodo.loc[caso3] = "Promedio S2 del cluster"

    # Metodo 2: sin S1_2026 pero con S2 propio
    caso2 = ~sin_movimiento & tiene_s2_propio & ~tiene_s1_2026
    proyeccion.loc[caso2] = df.loc[caso2, "base_propia_S2"]
    metodo.loc[caso2] = "Promedio S2 propio"

    # Metodo 1 (preferente): vendio en el semestre base
    caso1 = ~sin_movimiento & tiene_s1_2026
    proyeccion.loc[caso1] = (
        df.loc[caso1, col_base] * df.loc[caso1, "factor_cluster_S2"]
    )
    metodo.loc[caso1] = f"{sem_base} x factor cluster"

    # Metodo 4: sin ninguna venta -> el modelo no proyecta
    proyeccion.loc[sin_movimiento] = 0.0
    metodo.loc[sin_movimiento] = "Sin historial - ver hoja Sugerido_Lanzamientos"

    # Los nombres de estas columnas son GENERICOS a proposito. Antes llevaban el
    # semestre incrustado ('unidades_sugeridas_semestre'); al volverse dinamico el
    # calendario, ese nombre habria mentido en cuanto el objetivo pasara a ser
    # otro semestre. El semestre concreto queda en su propia columna.
    df["semestre_objetivo"] = sem_objetivo
    df["proyeccion_semestre"] = proyeccion.round(2)
    df["metodo_proyeccion"] = metodo

    # Se redondea hacia arriba: en compra semestral es preferible un excedente
    # marginal a un quiebre de stock que cuesta la venta completa.
    df["unidades_sugeridas_semestre"] = np.ceil(df["proyeccion_semestre"]).astype("int64")
    df["ingreso_proyectado_semestre"] = (
        df["unidades_sugeridas_semestre"] * df["PVP"]
    ).round(2)

    utils.log("Distribucion de metodos de proyeccion:")
    for nombre, n in df["metodo_proyeccion"].value_counts().items():
        utils.log(f"  {nombre}: {n} registros ({utils.porcentaje(n, len(df))}%)")
    utils.log(f"Unidades sugeridas totales {cfg.SEMESTRE_OBJETIVO}: "
              f"{utils.formatear_miles(df['unidades_sugeridas_semestre'].sum())}")

    return df


# ---------------------------------------------------------------------------
# 5. VISUALIZACIONES
# ---------------------------------------------------------------------------


def graficar_pca(df: pd.DataFrame, guardar=True):
    """Dispersion de los productos en las dos primeras componentes principales."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    datos = df.dropna(subset=["pca_1", "pca_2"])
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=datos, x="pca_1", y="pca_2", hue="Nombre_Cluster",
                    palette="tab10", s=45, alpha=0.75, ax=ax)
    ax.set_title("Clusters de producto en el espacio PCA (2 primeras componentes)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    if guardar:
        utils.guardar_figura(fig, "05_scatter_pca_clusters")
    return fig


def graficar_tsne(df: pd.DataFrame, matriz_pca: np.ndarray, guardar=True):
    """Proyeccion t-SNE, util para ver separaciones que el PCA lineal esconde.

    t-SNE se corre sobre la salida del PCA (no sobre los datos crudos): es la
    practica recomendada porque acelera el calculo y filtra ruido.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    utils.log("Calculando t-SNE (puede tardar algunos segundos) ...")
    perplejidad = min(30, max(5, (len(matriz_pca) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplejidad, init="pca",
                learning_rate="auto", random_state=cfg.SEMILLA)
    coordenadas = tsne.fit_transform(matriz_pca)

    # El orden de matriz_pca corresponde a los registros con movimiento
    datos = df.dropna(subset=["pca_1", "pca_2"]).copy()
    if len(datos) != len(coordenadas):
        utils.log("Desalineacion entre t-SNE y el dataset: se omite la figura",
                  nivel="ALERTA")
        return None
    datos["tsne_1"], datos["tsne_2"] = coordenadas[:, 0], coordenadas[:, 1]

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=datos, x="tsne_1", y="tsne_2", hue="Nombre_Cluster",
                    palette="tab10", s=45, alpha=0.75, ax=ax)
    ax.set_title(f"Proyeccion t-SNE de los clusters (perplejidad={perplejidad})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    if guardar:
        utils.guardar_figura(fig, "06_tsne_clusters")
    return fig


def graficar_perfil_clusters(df: pd.DataFrame, guardar=True):
    """Barras apiladas: composicion de cada cluster por subcategoria y canal."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for ax, columna, titulo in zip(
        axes,
        ["Subcategoria", "Canal", "Prioridad_Compra"],
        ["Composicion por subcategoria", "Composicion por canal",
         "Composicion por prioridad de compra"],
    ):
        tabla = pd.crosstab(df["Cluster"], df[columna], normalize="index") * 100
        tabla.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8)
        ax.set_title(titulo)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("% de registros")
        ax.legend(fontsize=7, bbox_to_anchor=(1.0, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Perfil de composicion de cada cluster", fontsize=13, fontweight="bold")
    if guardar:
        utils.guardar_figura(fig, "07_perfil_clusters")
    return fig


def graficar_metricas_por_cluster(perfiles: pd.DataFrame, guardar=True):
    """Compara las metricas clave entre clusters en un panel de 4 graficas."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    panel = [
        ("promedio_mensual_prom", "Promedio mensual de unidades", "steelblue"),
        ("cobertura_meses_prom", "Cobertura de meses (%)", "seagreen"),
        ("tendencia_prom", "Tendencia (unidades/mes)", "indianred"),
        ("variabilidad_prom", "Variabilidad (coef. de variacion)", "goldenrod"),
    ]

    for ax, (columna, titulo, color) in zip(axes.ravel(), panel):
        ax.bar(perfiles["Cluster"].astype(str), perfiles[columna], color=color)
        ax.set_title(titulo)
        ax.set_xlabel("Cluster")
        ax.axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Metricas promedio por cluster", fontsize=13, fontweight="bold")
    if guardar:
        utils.guardar_figura(fig, "08_metricas_por_cluster")
    return fig


def tabla_resumen(perfiles: pd.DataFrame) -> pd.DataFrame:
    """Version compacta de los perfiles, para presentar al negocio."""
    columnas = {
        "Cluster": "Cluster",
        "Nombre_Cluster": "Nombre",
        "n_registros": "Registros",
        "moda_Subcategoria": "Subcategoria dominante",
        "moda_Marca": "Marca dominante",
        "moda_Canal": "Canal dominante",
        "promedio_mensual_prom": "Prom. mensual (und)",
        "cobertura_meses_prom": "Cobertura (%)",
        "tendencia_prom": "Tendencia (und/mes)",
        "variabilidad_prom": "Variabilidad",
        "ingreso_estimado_suma": "Ingreso historico estimado",
    }
    existentes = {k: v for k, v in columnas.items() if k in perfiles.columns}
    return perfiles[list(existentes)].rename(columns=existentes)


# ---------------------------------------------------------------------------
# PIPELINE DE EVALUACION
# ---------------------------------------------------------------------------


def ejecutar(resultado=None, guardar: bool = True):
    """Punto de entrada del paso 4: perfila, nombra, prioriza y proyecta.

    Returns:
        (df_final, perfiles) con todo lo necesario para el Excel y el dashboard.
    """
    utils.titulo("PASO 4 - INTERPRETACION Y RESULTADOS")

    if resultado is None:
        resultado = cl.ejecutar()

    df = resultado.df if hasattr(resultado, "df") else resultado

    perfiles = perfilar_clusters(df)
    df, perfiles = asignar_nombres(df, perfiles)
    df = asignar_prioridad(df)
    df = proyectar_semestre_objetivo(df)

    # El nombre y la prioridad pueden cambiar la lectura del perfil, asi que la
    # tabla se recalcula al final para que quede consistente con el dataset.
    perfiles = perfilar_clusters(df)
    df, perfiles = asignar_nombres(df, perfiles)
    perfiles = perfiles.merge(
        df.groupby("Cluster")[["unidades_sugeridas_semestre"]].sum()
          .rename(columns={"unidades_sugeridas_semestre": "unidades_sugeridas_semestre_suma"}),
        on="Cluster", how="left",
    )
    perfiles = perfiles.merge(
        df.groupby("Cluster")["Prioridad_Compra"]
          .agg(lambda s: utils.moda_segura(s)).rename("prioridad_dominante"),
        on="Cluster", how="left",
    )

    utils.titulo("TABLA RESUMEN DE CLUSTERS")
    print(tabla_resumen(perfiles).to_string(index=False))

    if guardar:
        cfg.crear_directorios()
        df.to_parquet(cfg.ARCHIVO_CLUSTERS, index=False)
        perfiles.to_csv(cfg.DIR_OUTPUT_DATA / "perfiles_cluster.csv", index=False)
        utils.log(f"Resultados guardados en {cfg.ARCHIVO_CLUSTERS}")

    return df, perfiles


if __name__ == "__main__":
    ejecutar()
