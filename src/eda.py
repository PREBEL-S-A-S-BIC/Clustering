# -*- coding: utf-8 -*-
"""
eda.py - Analisis exploratorio de datos.

Reune las funciones de graficacion que usa el notebook 01_EDA.ipynb, de modo
que el notebook quede limpio (una llamada por celda) y las mismas figuras se
puedan regenerar desde consola con:

    python -m src.eda

Todas las figuras se guardan en outputs/figures/ con prefijo numerico para que
queden ordenadas por paso del analisis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import feature_engineering as fe
from . import ingesta, utils


# ---------------------------------------------------------------------------
# 1. DISTRIBUCION DE VENTAS POR DIMENSION
# ---------------------------------------------------------------------------


def graficar_distribucion_por_dimension(df: pd.DataFrame, guardar=True):
    """Unidades vendidas agregadas por canal, marca, subcategoria y grupo.

    'Categoria' se omite a proposito: en esta fuente es constante
    ('Maquillaje'), asi que la jerarquia informativa arranca en Subcategoria.
    """
    import matplotlib.pyplot as plt

    meses = [m for m in cfg.MESES if m in df.columns]
    datos = df.copy()
    datos["total"] = datos[meses].sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    panel = [
        ("Canal", "Unidades vendidas por canal", None),
        ("Marca", "Unidades vendidas por marca", None),
        ("Subcategoria", "Unidades vendidas por subcategoria", None),
        ("Grupo", "Top 15 grupos por unidades vendidas", 15),
    ]

    for ax, (columna, titulo, top) in zip(axes.ravel(), panel):
        serie = datos.groupby(columna)["total"].sum().sort_values(ascending=False)
        if top:
            serie = serie.head(top)
        serie.sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(titulo)
        ax.set_xlabel("Unidades vendidas (30 meses)")
        ax.set_ylabel("")
        # Etiqueta de valor al final de cada barra: evita tener que estimar
        for y, valor in enumerate(serie.sort_values()):
            ax.text(valor, y, f" {utils.formatear_miles(valor)}", va="center", fontsize=8)

    fig.suptitle("Distribucion de ventas por dimension", fontsize=14, fontweight="bold")
    if guardar:
        utils.guardar_figura(fig, "01_distribucion_por_dimension")
    return fig


# ---------------------------------------------------------------------------
# 2. EVOLUCION TEMPORAL
# ---------------------------------------------------------------------------


def graficar_evolucion_mensual(df: pd.DataFrame, guardar=True):
    """Serie de tiempo de los 30 meses: total y desglose por canal."""
    import matplotlib.pyplot as plt

    meses = [m for m in cfg.MESES if m in df.columns]
    total = df[meses].sum()
    por_canal = df.groupby("Canal")[meses].sum().T

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # --- Total mensual con marcas de fin de semestre ---------------------
    axes[0].plot(total.index, total.values, marker="o", linewidth=2, color="darkblue")
    axes[0].fill_between(range(len(total)), total.values, alpha=0.15, color="darkblue")
    axes[0].set_title("Evolucion mensual de unidades vendidas (total 3 canales)")
    axes[0].set_ylabel("Unidades")
    for semestre, meses_sem in cfg.SEMESTRES.items():
        ultimo = meses_sem[-1]
        if ultimo in total.index:
            posicion = list(total.index).index(ultimo)
            axes[0].axvline(posicion + 0.5, color="gray", linestyle=":", linewidth=1)
            axes[0].text(posicion, total.max() * 1.02, semestre,
                         fontsize=8, ha="right", color="gray")

    # --- Desglose por canal ----------------------------------------------
    for canal in por_canal.columns:
        axes[1].plot(por_canal.index, por_canal[canal], marker="o",
                     linewidth=1.8, label=canal)
    axes[1].set_title("Evolucion mensual por canal")
    axes[1].set_ylabel("Unidades")
    axes[1].set_xlabel("Mes")
    axes[1].legend(title="Canal")
    axes[1].tick_params(axis="x", rotation=90)

    if guardar:
        utils.guardar_figura(fig, "02_evolucion_mensual")
    return fig


def graficar_heatmap_productos(df: pd.DataFrame, top=50, guardar=True):
    """Mapa de calor producto x mes para los productos de mayor venta.

    Se aplica escala logaritmica al color: sin ella los 2 o 3 productos mas
    vendidos aplastan visualmente al resto y el mapa no muestra el patron
    temporal, que es justamente lo que se quiere leer.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    meses = [m for m in cfg.MESES if m in df.columns]
    datos = df.copy()
    datos["total"] = datos[meses].sum(axis=1)
    datos["etiqueta"] = (
        datos["Descripcion"].str.slice(0, 34) + " | " + datos["Canal"].str.slice(0, 4)
    )

    top_n = datos.nlargest(top, "total")
    matriz = top_n.set_index("etiqueta")[meses]

    fig, ax = plt.subplots(figsize=(15, max(9, top * 0.22)))
    sns.heatmap(np.log1p(matriz), cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "log(1 + unidades)"})
    ax.set_title(f"Mapa de calor: top {top} productos x mes (escala logaritmica)")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Producto | Canal")
    if guardar:
        utils.guardar_figura(fig, "03_heatmap_productos")
    return fig


# ---------------------------------------------------------------------------
# 3. DISTRIBUCION DE LAS FEATURES
# ---------------------------------------------------------------------------


def graficar_cobertura(features: pd.DataFrame, guardar=True):
    """Histograma de cobertura: distingue venta permanente de esporadica."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].hist(features["cobertura_meses"], bins=30,
                 color="teal", edgecolor="white")
    axes[0].axvline(cfg.UMBRAL_COBERTURA_ALTA, color="green", linestyle="--",
                    label=f"Umbral alta ({cfg.UMBRAL_COBERTURA_ALTA}%)")
    axes[0].axvline(cfg.UMBRAL_COBERTURA_BAJA, color="red", linestyle="--",
                    label=f"Umbral baja ({cfg.UMBRAL_COBERTURA_BAJA}%)")
    axes[0].set_title("Distribucion de cobertura de meses")
    axes[0].set_xlabel("% de los 30 meses con venta")
    axes[0].set_ylabel("Cantidad de registros producto x canal")
    axes[0].legend()

    # Tramos de cobertura: lectura directa para el negocio
    tramos = pd.cut(
        features["cobertura_meses"],
        bins=[-0.1, 0, 20, 40, 70, 90, 100],
        labels=["Sin venta", "1-20%", "20-40%", "40-70%", "70-90%", "90-100%"],
    ).value_counts().sort_index()

    tramos.plot(kind="bar", ax=axes[1], color="darkslategray")
    axes[1].set_title("Registros por tramo de cobertura")
    axes[1].set_xlabel("Tramo de cobertura")
    axes[1].set_ylabel("Cantidad de registros")
    axes[1].tick_params(axis="x", rotation=30)
    for x, valor in enumerate(tramos):
        axes[1].text(x, valor, str(valor), ha="center", va="bottom", fontsize=9)

    if guardar:
        utils.guardar_figura(fig, "04a_distribucion_cobertura")
    return fig


def graficar_boxplots(features: pd.DataFrame, guardar=True):
    """Boxplot de total_unidades por subcategoria y por canal.

    Eje logaritmico: la distribucion de unidades abarca varios ordenes de
    magnitud y en escala lineal todas las cajas quedarian pegadas a cero.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    datos = features.loc[features["total_unidades"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    orden_sub = (datos.groupby("Subcategoria")["total_unidades"].median()
                      .sort_values(ascending=False).index)
    sns.boxplot(data=datos, x="Subcategoria", y="total_unidades",
                order=orden_sub, ax=axes[0], hue="Subcategoria",
                palette="Set2", legend=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Total de unidades por subcategoria (escala log)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(data=datos, x="Canal", y="total_unidades",
                ax=axes[1], hue="Canal", palette="Set3", legend=False)
    axes[1].set_yscale("log")
    axes[1].set_title("Total de unidades por canal (escala log)")

    fig.suptitle("Dispersion del volumen de venta (solo registros con venta)",
                 fontsize=13, fontweight="bold")
    if guardar:
        utils.guardar_figura(fig, "04b_boxplots_volumen")
    return fig


def graficar_tendencias(features: pd.DataFrame, guardar=True):
    """Compara productos en crecimiento vs. en declive."""
    import matplotlib.pyplot as plt

    datos = features.loc[features["total_unidades"] > 0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Conteo por signo de tendencia -----------------------------------
    signo = pd.cut(
        datos["tendencia"],
        bins=[-np.inf, cfg.UMBRAL_TENDENCIA_NEGATIVA, 1, np.inf],
        labels=["En declive", "Estable", "En crecimiento"],
    ).value_counts().reindex(["En declive", "Estable", "En crecimiento"])

    axes[0].bar(signo.index, signo.values,
                color=["indianred", "goldenrod", "seagreen"])
    axes[0].set_title("Registros por dinamica de venta")
    axes[0].set_ylabel("Cantidad de registros")
    for x, valor in enumerate(signo.values):
        axes[0].text(x, valor, str(valor), ha="center", va="bottom")

    # --- Top 15 en crecimiento -------------------------------------------
    top_sube = datos.nlargest(15, "tendencia")
    etiquetas = (top_sube["Descripcion"].str.slice(0, 28) + " | "
                 + top_sube["Canal"].str.slice(0, 4))
    axes[1].barh(etiquetas, top_sube["tendencia"], color="seagreen")
    axes[1].invert_yaxis()
    axes[1].set_title("Top 15 en crecimiento")
    axes[1].set_xlabel("Tendencia (und/mes)")
    axes[1].tick_params(axis="y", labelsize=7)

    # --- Top 15 en declive -----------------------------------------------
    top_baja = datos.nsmallest(15, "tendencia")
    etiquetas = (top_baja["Descripcion"].str.slice(0, 28) + " | "
                 + top_baja["Canal"].str.slice(0, 4))
    axes[2].barh(etiquetas, top_baja["tendencia"], color="indianred")
    axes[2].invert_yaxis()
    axes[2].set_title("Top 15 en declive")
    axes[2].set_xlabel("Tendencia (und/mes)")
    axes[2].tick_params(axis="y", labelsize=7)

    if guardar:
        utils.guardar_figura(fig, "04c_tendencias")
    return fig


def graficar_correlaciones(features: pd.DataFrame, guardar=True):
    """Matriz de correlacion entre las features numericas del modelo.

    Sirve para detectar redundancia: si dos variables correlacionan casi 1,
    aportan la misma informacion y el PCA las va a colapsar de todas formas.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    columnas = cfg.FEATURES_NUMERICAS + ["ingreso_estimado", "meses_activos", "PVP"]
    existentes = [c for c in columnas if c in features.columns]
    matriz = features[existentes].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9.5, 8))
    mascara = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mascara, ax=ax, square=True, linewidths=0.5,
                cbar_kws={"label": "Correlacion de Spearman"})
    ax.set_title("Correlacion entre las variables numericas\n"
                 "(Spearman: robusta a distribuciones sesgadas)")
    if guardar:
        utils.guardar_figura(fig, "04d_correlaciones")
    return fig


# ---------------------------------------------------------------------------
# RESUMENES EN TEXTO
# ---------------------------------------------------------------------------


def resumen_eda(df: pd.DataFrame, features: pd.DataFrame) -> None:
    """Imprime los hallazgos numericos principales del EDA."""
    meses = [m for m in cfg.MESES if m in df.columns]

    utils.titulo("HALLAZGOS DEL EDA")
    total = int(df[meses].to_numpy().sum())
    utils.log(f"Unidades vendidas en 30 meses: {utils.formatear_miles(total)}")
    utils.log(f"Registros producto x canal: {len(df):,} "
              f"({df['Material'].nunique()} productos x {df['Canal'].nunique()} canales)")

    print("\nParticipacion por canal:")
    participacion = df.groupby("Canal")[meses].sum().sum(axis=1).sort_values(ascending=False)
    for canal, unidades in participacion.items():
        print(f"  {canal:<12} {utils.formatear_miles(unidades):>12} "
              f"({utils.porcentaje(unidades, total)}%)")

    print("\nParticipacion por marca:")
    por_marca = df.groupby("Marca")[meses].sum().sum(axis=1).sort_values(ascending=False)
    for marca, unidades in por_marca.items():
        print(f"  {marca:<12} {utils.formatear_miles(unidades):>12} "
              f"({utils.porcentaje(unidades, total)}%)")

    print("\nConcentracion de la venta (regla de Pareto):")
    ordenado = features.sort_values("total_unidades", ascending=False)
    acumulado = ordenado["total_unidades"].cumsum() / ordenado["total_unidades"].sum()
    n_80 = int((acumulado <= 0.80).sum()) + 1
    print(f"  {n_80} registros ({utils.porcentaje(n_80, len(features))}%) "
          f"concentran el 80% de las unidades")

    print("\nDinamica de venta:")
    con_venta = features.loc[features["total_unidades"] > 0]
    print(f"  En crecimiento (tendencia > 0): {int((con_venta['tendencia'] > 0).sum())}")
    print(f"  En declive (tendencia < 0):     {int((con_venta['tendencia'] < 0).sum())}")
    print(f"  Sin venta en 30 meses:          {int((features['total_unidades'] == 0).sum())}")


def ejecutar(guardar=True):
    """Genera todas las figuras del EDA y devuelve (maestro, features)."""
    utils.titulo("EDA - ANALISIS EXPLORATORIO")
    utils.aplicar_estilo_graficas()

    maestro = ingesta.cargar_maestro()
    features = fe.cargar_features()

    graficar_distribucion_por_dimension(maestro, guardar)
    graficar_evolucion_mensual(maestro, guardar)
    graficar_heatmap_productos(maestro, top=50, guardar=guardar)
    graficar_cobertura(features, guardar)
    graficar_boxplots(features, guardar)
    graficar_tendencias(features, guardar)
    graficar_correlaciones(features, guardar)

    resumen_eda(maestro, features)

    import matplotlib.pyplot as plt
    plt.close("all")
    return maestro, features


if __name__ == "__main__":
    ejecutar()
