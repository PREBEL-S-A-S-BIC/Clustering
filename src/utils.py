# -*- coding: utf-8 -*-
"""
utils.py - Utilidades comunes y construccion del entregable final en Excel.

Contiene:
    1. Helpers genericos (logging, normalizacion de texto, guardado de figuras).
    2. La funcion `exportar_excel_final()` que arma Cosnova_Clusters_Final.xlsx
       con las 4 hojas que pide el negocio.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg

# ---------------------------------------------------------------------------
# 1. HELPERS GENERICOS
# ---------------------------------------------------------------------------


def log(mensaje: str, nivel: str = "INFO") -> None:
    """Imprime un mensaje de avance con formato uniforme.

    Se usa un print simple en lugar de `logging` porque el proyecto se ejecuta
    principalmente desde notebooks y consola, donde print es mas legible.
    """
    print(f"[{nivel}] {mensaje}")


def titulo(texto: str, ancho: int = 78) -> None:
    """Imprime un separador de seccion para que los logs sean navegables."""
    print("\n" + "=" * ancho)
    print(texto)
    print("=" * ancho)


def info_interprete() -> dict:
    """Informa con que interprete de Python se esta ejecutando el proyecto.

    Sirve para detectar a tiempo el caso incomodo: ejecutar con el Python del
    sistema en lugar del entorno virtual. Hoy ambos tienen las mismas librerias,
    asi que el proyecto correria igual y el error pasaria inadvertido, hasta que
    las versiones diverjan y algo se comporte distinto sin explicacion.

    Returns:
        dict con la ruta del ejecutable, si esta dentro de un entorno virtual y
        el nombre del entorno.
    """
    import sys

    # En un entorno virtual, sys.prefix apunta al entorno y sys.base_prefix al
    # Python que lo creo. Si son iguales, no hay entorno activo.
    en_entorno = sys.prefix != sys.base_prefix

    return {
        "ejecutable": sys.executable,
        "en_entorno_virtual": en_entorno,
        "entorno": Path(sys.prefix).name if en_entorno else "Python del sistema",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}."
                   f"{sys.version_info.micro}",
    }


def log_interprete() -> dict:
    """Imprime en consola que interprete se esta usando. Devuelve la info."""
    info = info_interprete()
    etiqueta = (f"entorno virtual '{info['entorno']}'" if info["en_entorno_virtual"]
                else "PYTHON DEL SISTEMA")
    log(f"Interprete: {info['ejecutable']}")
    log(f"Python {info['version']} | {etiqueta}")

    if not info["en_entorno_virtual"]:
        log("No estas usando el entorno virtual del proyecto. Funciona igual "
            "porque el sistema tiene las mismas librerias, pero las versiones "
            "pueden diverger con el tiempo.", nivel="ALERTA")
        log(r"  Para usarlo: C:\venvs\cosnova\Scripts\python.exe run_pipeline.py",
            nivel="ALERTA")

    return info


def quitar_acentos(texto: str) -> str:
    """Elimina tildes y diacriticos de un texto ('Descripción' -> 'Descripcion')."""
    if not isinstance(texto, str):
        return texto
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def normalizar_nombre_columna(nombre) -> str:
    """Normaliza un encabezado del Excel a un nombre de columna manejable.

    Quita espacios sobrantes (la fuente trae 'Material ' y 'Descripción ' con
    espacio final) y elimina las tildes para evitar problemas de encoding al
    referenciar columnas desde codigo.
    """
    if not isinstance(nombre, str):
        return nombre
    limpio = re.sub(r"\s+", " ", nombre).strip()
    return quitar_acentos(limpio)


def normalizar_texto(valor) -> str:
    """Normaliza un valor de celda de texto para comparaciones robustas.

    Pasa a minusculas, quita tildes y colapsa espacios. Se usa para detectar
    los 'Ex article' de la fuente, que vienen escritos de tres formas distintas.
    """
    if pd.isna(valor):
        return ""
    return re.sub(r"\s+", " ", quitar_acentos(str(valor)).strip().lower())


def limpiar_categoria(valor) -> str:
    """Estandariza un valor categorico conservando su forma legible.

    Aplica Capitalizacion consistente y unifica variantes que en la fuente
    difieren solo por tilde (por ejemplo 'Líquido' y 'Liquido' son el mismo
    tipo de producto y deben quedar en una sola categoria).
    """
    if pd.isna(valor) or str(valor).strip() == "":
        return "Sin dato"
    limpio = re.sub(r"\s+", " ", str(valor)).strip()
    # Se usa la version sin tildes como forma canonica para que 'Líquido' y
    # 'Liquido' colapsen en un unico nivel.
    return quitar_acentos(limpio).capitalize()


def unificar_grupo(valor) -> str:
    """Unifica los sinonimos singular/plural de 'Grupo'.

    Se aplica en dos lugares y por eso vive aqui: en la ingesta (para limpiar el
    maestro) y al resolver lo que el usuario carga en un Excel de lanzamientos.
    Si solo se aplicara en la ingesta, alguien que escribiera 'Pestañinas' en su
    archivo no encontraria los comparables de 'Pestañina'.
    """
    limpio = limpiar_categoria(valor)
    return cfg.SINONIMOS_GRUPO.get(limpio, limpio)


def guardar_figura(fig, nombre: str) -> str:
    """Guarda una figura de matplotlib en outputs/figures/ y devuelve su ruta."""
    cfg.crear_directorios()
    ruta = cfg.DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, dpi=cfg.DPI_FIGURAS, bbox_inches="tight")
    log(f"Figura guardada: {ruta.name}")
    return str(ruta)


def aplicar_estilo_graficas() -> None:
    """Fija un estilo visual unico para todas las figuras del proyecto."""
    import matplotlib

    matplotlib.use("Agg", force=False)  # backend sin ventana: sirve en scripts
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = cfg.DPI_FIGURAS
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.autolayout"] = True


def moda_segura(serie: pd.Series, defecto: str = "Sin dato") -> str:
    """Devuelve el valor mas frecuente de una serie; tolera series vacias.

    `Series.mode()` devuelve un vector vacio cuando todo es nulo, lo que
    rompe el perfilamiento; esta version siempre devuelve un string.
    """
    serie = serie.dropna()
    if serie.empty:
        return defecto
    modas = serie.mode()
    return str(modas.iloc[0]) if len(modas) else defecto


def porcentaje(parte: float, total: float) -> float:
    """Division segura expresada en porcentaje (evita ZeroDivisionError)."""
    return 0.0 if total == 0 else round(100.0 * parte / total, 2)


def columnas_semestre(prefijo: str = "unidades") -> list[str]:
    """Devuelve los nombres de las columnas semestrales, en orden cronologico."""
    return [f"{prefijo}_{s}" for s in cfg.SEMESTRES]


def ajustar_ancho_columnas(writer, hoja: str, df: pd.DataFrame, maximo: int = 45) -> None:
    """Ajusta el ancho de las columnas de una hoja de Excel al contenido.

    Mejora bastante la legibilidad del entregable: sin esto las descripciones
    largas quedan cortadas y los encabezados ilegibles.
    """
    ws = writer.sheets[hoja]
    for i, columna in enumerate(df.columns, start=1):
        # Se mide el encabezado y una muestra de valores (no todo el df, por
        # rendimiento) para estimar el ancho necesario.
        muestra = df[columna].astype(str).head(200)
        largo_datos = muestra.map(len).max() if len(muestra) else 0
        ancho = min(max(len(str(columna)) + 2, int(largo_datos) + 2), maximo)
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = ancho
    ws.freeze_panes = "A2"  # deja el encabezado fijo al hacer scroll


# ---------------------------------------------------------------------------
# 2. HOJAS DEL ENTREGABLE FINAL
# ---------------------------------------------------------------------------


def construir_hoja_productos(df: pd.DataFrame) -> pd.DataFrame:
    """Hoja 'Productos_Clasificados': dataset completo con cluster y prioridad."""
    columnas = [
        "Material", "EAN/UPC-Code", "Descripcion", "Marca", "Canal",
        "Status", "Tipo Portafolio", "Categoria", "Subcategoria", "Grupo",
        "Tono", "Tipo", "PVP",
        "total_unidades", "promedio_mensual", "cobertura_meses", "tendencia",
        "estacionalidad", "variabilidad", "ingreso_estimado",
        "mejor_semestre", "meses_activos",
        *columnas_semestre("unidades"),
        "Cluster", "Nombre_Cluster", "Prioridad_Compra",
    ]
    existentes = [c for c in columnas if c in df.columns]
    salida = df[existentes].copy()
    return salida.sort_values(
        ["Canal", "Cluster", "total_unidades"], ascending=[True, True, False]
    )


def construir_hoja_perfiles(perfiles: pd.DataFrame) -> pd.DataFrame:
    """Hoja 'Perfiles_Cluster': resumen estadistico por cluster."""
    return perfiles.copy()


def construir_hoja_ventas_semestre(df: pd.DataFrame) -> pd.DataFrame:
    """Hoja 'Ventas_por_Semestre': tabla producto x semestre con totales.

    Cada fila es un producto-canal (la unidad de analisis del modelo) y las
    columnas son los 5 semestres con historia, mas el total.
    """
    cols_sem = [c for c in columnas_semestre("unidades") if c in df.columns]
    llaves = ["Material", "Descripcion", "Marca", "Canal",
              "Subcategoria", "Grupo", "Cluster", "Nombre_Cluster"]
    llaves = [c for c in llaves if c in df.columns]

    pivote = df[llaves + cols_sem].copy()
    pivote["Total_30_meses"] = pivote[cols_sem].sum(axis=1)

    # Fila de totales al final para lectura rapida del negocio
    totales = {c: pivote[c].sum() for c in cols_sem + ["Total_30_meses"]}
    fila = {c: "" for c in llaves}
    fila["Material"] = "TOTAL"
    fila.update(totales)

    return pd.concat(
        [pivote.sort_values("Total_30_meses", ascending=False),
         pd.DataFrame([fila])],
        ignore_index=True,
    )


def construir_hoja_recomendacion(df: pd.DataFrame) -> pd.DataFrame:
    """Hoja 'Recomendacion_Compra': proyeccion sugerida para S2_2026.

    La proyeccion la calcula `evaluation.proyectar_semestre_objetivo`; aqui
    solo se selecciona y ordena la informacion que el comprador necesita ver.
    """
    columnas = [
        "Material", "EAN/UPC-Code", "Descripcion", "Marca", "Canal",
        "Subcategoria", "Grupo", "Tono", "PVP", "Status", "Tipo Portafolio",
        "Cluster", "Nombre_Cluster", "Prioridad_Compra",
        "unidades_S2_2024", "unidades_S2_2025", "unidades_S1_2026",
        "base_propia_S2", "factor_cluster_S2", "tendencia",
        "proyeccion_semestre", "unidades_sugeridas_semestre",
        "ingreso_proyectado_semestre", "metodo_proyeccion",
    ]
    existentes = [c for c in columnas if c in df.columns]
    salida = df[existentes].copy()

    # Orden de trabajo del comprador: canal, luego prioridad, luego volumen.
    orden_prioridad = {"Alta": 0, "Media": 1, "Baja": 2, "Sin Movimiento": 3}
    salida["_orden"] = salida["Prioridad_Compra"].map(orden_prioridad).fillna(9)
    salida = salida.sort_values(
        ["Canal", "_orden", "unidades_sugeridas_semestre"],
        ascending=[True, True, False],
    ).drop(columns="_orden")
    return salida


def construir_hoja_lanzamientos(sugeridos: pd.DataFrame) -> pd.DataFrame:
    """Hoja 'Sugerido_Lanzamientos': cantidad estimada para productos sin historial.

    Estos productos no tienen historia de venta, asi que el modelo de clustering no
    puede proyectarlos. La cifra viene del modelo de atributos (`lanzamientos.py`),
    que estima desde canal, marca, subcategoria, grupo, tipo, tono, PVP y mes de
    lanzamiento.
    """
    salida = sugeridos.copy()
    # Se ordena por lo que mas pesa en la compra, marcando primero lo que necesita
    # validacion comercial.
    return salida.sort_values(
        ["requiere_revision", "Canal", "unidades_sugeridas_lanzamiento"],
        ascending=[False, True, False],
    )


def exportar_excel_final(
    df: pd.DataFrame | None = None,
    perfiles: pd.DataFrame | None = None,
    sugeridos_lanzamiento: pd.DataFrame | None = None,
    ruta_salida=None,
) -> str:
    """Arma y guarda Cosnova_Clusters_Final.xlsx.

    Args:
        df: dataset final con cluster, nombre, prioridad y proyeccion.
            Si es None se lee de data/output/clusters.parquet.
        perfiles: resumen por cluster. Si es None se recalcula.
        sugeridos_lanzamiento: sugerido para productos sin historial. Si es None se
            intenta leer de data/output/sugerido_lanzamientos.parquet; si tampoco
            existe, la hoja simplemente se omite.
        ruta_salida: ruta alternativa del Excel (por defecto la de config).

    Returns:
        La ruta del archivo generado.
    """
    from . import evaluation  # import local para evitar dependencia circular

    cfg.crear_directorios()

    if df is None:
        log("Leyendo resultados desde data/output/clusters.parquet ...")
        df = pd.read_parquet(cfg.ARCHIVO_CLUSTERS)
    if perfiles is None:
        perfiles = evaluation.perfilar_clusters(df)

    if sugeridos_lanzamiento is None:
        ruta_sugeridos = cfg.DIR_OUTPUT_DATA / "sugerido_lanzamientos.parquet"
        if ruta_sugeridos.exists():
            sugeridos_lanzamiento = pd.read_parquet(ruta_sugeridos)

    ruta = ruta_salida or cfg.ARCHIVO_EXCEL_FINAL

    hojas = {
        "Productos_Clasificados": construir_hoja_productos(df),
        "Perfiles_Cluster": construir_hoja_perfiles(perfiles),
        "Ventas_por_Semestre": construir_hoja_ventas_semestre(df),
        "Recomendacion_Compra": construir_hoja_recomendacion(df),
    }
    if sugeridos_lanzamiento is not None and len(sugeridos_lanzamiento):
        hojas["Sugerido_Lanzamientos"] = construir_hoja_lanzamientos(sugeridos_lanzamiento)
    else:
        log("Sin sugeridos de lanzamiento: la hoja 'Sugerido_Lanzamientos' se omite. "
            "Ejecuta 'python -m src.lanzamientos' para generarla.", nivel="ALERTA")

    titulo("EXPORTANDO ENTREGABLE FINAL")
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        for nombre_hoja, tabla in hojas.items():
            tabla.to_excel(writer, sheet_name=nombre_hoja, index=False)
            ajustar_ancho_columnas(writer, nombre_hoja, tabla)
            log(f"Hoja '{nombre_hoja}': {len(tabla):,} filas x {tabla.shape[1]} columnas")

    log(f"Excel final generado: {ruta}")
    return str(ruta)


def resumen_dataframe(df: pd.DataFrame, nombre: str = "DataFrame") -> None:
    """Imprime un diagnostico rapido de un DataFrame (forma, nulos, memoria)."""
    titulo(f"RESUMEN: {nombre}")
    log(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
    log(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        log("Sin valores nulos")
    else:
        log("Columnas con nulos:")
        for col, n in nulos.items():
            print(f"       - {col}: {n} ({porcentaje(n, len(df))}%)")


def formatear_miles(valor) -> str:
    """Formatea un numero con separador de miles para mostrarlo en pantalla."""
    if pd.isna(valor):
        return "-"
    return f"{valor:,.0f}".replace(",", ".")
