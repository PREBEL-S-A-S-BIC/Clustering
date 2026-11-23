# -*- coding: utf-8 -*-
"""
ingesta.py - Lectura, limpieza y unificacion de las hojas de canal.

Que hace:
    1. Lee las hojas Exito / Farmatodo / Prosalon del Excel fuente.
    2. Normaliza encabezados y renombra los meses a formato 'ene-24'.
    3. Agrega la columna 'Canal' con el nombre de la hoja.
    4. Unifica todo en un DataFrame maestro (un registro = producto x canal).
    5. Excluye los articulos descontinuados (Status = 'Ex article').
    6. Valida duplicados y consistencia de datos.
    7. Guarda el resultado en data/processed/maestro.parquet.

Notas sobre la fuente real (verificadas contra el archivo):
    - El Excel tiene 6 hojas; 'BD', 'Ventas' y 'TD POS' son insumos previos del
      negocio y no se usan aqui.
    - Los encabezados de mes son fechas (datetime), no texto: hay que
      convertirlos a etiquetas 'ene-24'.
    - 'Status' trae tres escrituras distintas para el mismo concepto:
      'no change', 'new article', 'Ex artcile' (typo) y 'ex article'.
    - 'EAN/UPC-Code' tiene un duplicado real por hoja (mismo EAN en un material
      Catrice y en uno Essence). Por eso la clave del modelo es 'Material'.
    - 'Categoria' es constante ('Maquillaje'): la jerarquia util es
      Subcategoria -> Grupo -> Tono.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as cfg
from . import utils

# Mapa de numero de mes a la abreviatura en espanol usada en el proyecto
ABREV_MES = {
    1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
    7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic",
}


# ---------------------------------------------------------------------------
# NORMALIZACION DE ENCABEZADOS
# ---------------------------------------------------------------------------


def _etiqueta_mes(columna) -> str | None:
    """Convierte un encabezado de mes a la etiqueta 'ene-24'.

    Acepta los dos formatos que puede devolver pandas al leer el Excel:
    un Timestamp/date (caso real de este archivo) o un texto ya con formato
    'ene-24' (por si el archivo se re-guarda como texto en el futuro).
    Devuelve None si la columna no representa un mes.
    """
    # Caso 1: encabezado como fecha
    if isinstance(columna, (pd.Timestamp, dt.datetime, dt.date)):
        return f"{ABREV_MES[columna.month]}-{str(columna.year)[-2:]}"

    # Caso 2: encabezado como texto tipo 'ene-24' o 'ene-2024'
    if isinstance(columna, str):
        texto = utils.normalizar_texto(columna).replace(" ", "")
        for numero, abrev in ABREV_MES.items():
            for separador in ("-", "/", "."):
                if texto.startswith(f"{abrev}{separador}"):
                    anio = texto.split(separador, 1)[1]
                    if anio.isdigit():
                        return f"{abrev}-{anio[-2:]}"
        # Ultimo intento: dejar que pandas lo interprete como fecha
        try:
            fecha = pd.to_datetime(columna, errors="raise")
            return f"{ABREV_MES[fecha.month]}-{str(fecha.year)[-2:]}"
        except (ValueError, TypeError):
            return None
    return None


def normalizar_encabezados(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas: atributos sin tildes ni espacios, meses a 'ene-24'."""
    nuevos = {}
    for columna in df.columns:
        etiqueta = _etiqueta_mes(columna)
        nuevos[columna] = etiqueta if etiqueta else utils.normalizar_nombre_columna(columna)
    return df.rename(columns=nuevos)


# ---------------------------------------------------------------------------
# LECTURA DE HOJAS
# ---------------------------------------------------------------------------


def leer_hoja(nombre_hoja: str, ruta_excel=None) -> pd.DataFrame:
    """Lee una hoja de canal y devuelve el DataFrame ya normalizado.

    Se agrega la columna 'Canal' con el nombre de la hoja, que es lo que
    permite tratar el mismo producto en distintos canales como registros
    independientes (requerimiento del negocio).
    """
    ruta = ruta_excel or cfg.ARCHIVO_EXCEL
    utils.log(f"Leyendo hoja '{nombre_hoja}' ...")

    df = pd.read_excel(ruta, sheet_name=nombre_hoja, engine="openpyxl")
    df = normalizar_encabezados(df)
    df["Canal"] = nombre_hoja

    utils.log(f"  -> {len(df):,} filas, {df.shape[1]} columnas")
    return df


def leer_todas_las_hojas(ruta_excel=None) -> dict[str, pd.DataFrame]:
    """Lee las 3 hojas de canal y las devuelve en un diccionario."""
    return {hoja: leer_hoja(hoja, ruta_excel) for hoja in cfg.HOJAS_CANAL}


# ---------------------------------------------------------------------------
# LIMPIEZA
# ---------------------------------------------------------------------------


def detectar_meses(df: pd.DataFrame) -> list[str]:
    """Detecta del archivo TODAS las columnas que son meses de venta.

    No se filtra contra la lista de meses configurada: se lee lo que el Excel
    realmente trae. Asi, cuando el negocio agrega la venta POS de meses nuevos,
    el modelo los toma sin que haya que tocar el codigo.
    """
    meses = [c for c in df.columns if _es_etiqueta_mes(c)]
    return cfg.ordenar_meses(meses)


def _es_etiqueta_mes(columna) -> bool:
    """True si el nombre de columna tiene la forma 'ene-24'."""
    if not isinstance(columna, str) or "-" not in columna:
        return False
    abrev, _, anio = columna.partition("-")
    return abrev in ABREV_MES.values() and anio.isdigit() and len(anio) == 2


def verificar_meses(df: pd.DataFrame) -> list[str]:
    """Sincroniza el calendario del proyecto con el del archivo fuente.

    Detecta los meses que trae el Excel y actualiza `config` para que el resto
    del pipeline (features, semestres, proyeccion) trabaje sobre ese calendario.
    """
    presentes = detectar_meses(df)
    if not presentes:
        utils.log("No se detecto ninguna columna de mes en la fuente", nivel="ERROR")
        return []

    cambios = cfg.actualizar_calendario(presentes)

    utils.log(f"Meses de venta detectados: {len(presentes)} "
              f"({cambios['primer_mes']} a {cambios['ultimo_mes']})")

    if cambios["meses_agregados"]:
        utils.log(f"Meses NUEVOS respecto al calendario anterior: "
                  f"{cambios['meses_agregados']}")
    if cambios["meses_quitados"]:
        utils.log(f"Meses que ya no estan en la fuente: {cambios['meses_quitados']}",
                  nivel="ALERTA")

    utils.log(f"Semestres completos: {cambios['semestres_completos']}")
    utils.log(f"Semestre a planear: {cambios['semestre_objetivo']}")

    # Un semestre incompleto al final es normal (el anio en curso), pero conviene
    # decirlo: sus cifras no son comparables con las de un semestre cerrado.
    incompletos = [s for s, m in cfg.SEMESTRES.items() if len(m) < 6]
    if incompletos:
        utils.log(f"Semestres incompletos (no se usan para comparar): {incompletos}")

    return presentes


def limpiar_tipos(df: pd.DataFrame, meses: list[str]) -> pd.DataFrame:
    """Convierte los meses y el PVP a numerico y estandariza los textos."""
    df = df.copy()

    # Unidades: cualquier valor no numerico o vacio se interpreta como 0 venta.
    for mes in meses:
        df[mes] = pd.to_numeric(df[mes], errors="coerce").fillna(0)
        # No existen devoluciones en esta fuente; un negativo seria un error de
        # captura, asi que se acota a 0 para no distorsionar los promedios.
        df[mes] = df[mes].clip(lower=0).astype("int64")

    # PVP: precio de venta al publico, base del ingreso estimado.
    df["PVP"] = pd.to_numeric(df["PVP"], errors="coerce").fillna(0).astype("float64")

    # Textos descriptivos: se recortan espacios sobrantes.
    for columna in ("Descripcion", "Material", "EAN/UPC-Code", "Caracteristica"):
        if columna in df.columns:
            df[columna] = df[columna].astype(str).str.strip()

    # Categoricas del modelo: se unifican variantes con y sin tilde
    # ('Líquido' / 'Liquido' quedan como un solo nivel).
    for columna in ("Marca", "Categoria", "Subcategoria", "Grupo", "Tono",
                    "Tipo", "Tipo Portafolio", "Status"):
        if columna in df.columns:
            df[columna] = df[columna].map(utils.limpiar_categoria)

    # 'Grupo' ademas trae el mismo valor en singular y plural como si fueran
    # categorias distintas ('Pestanina' / 'Pestaninas'). Se unifican para no
    # fragmentar las cohortes de productos comparables.
    if "Grupo" in df.columns:
        antes = df["Grupo"].nunique()
        df["Grupo"] = df["Grupo"].map(utils.unificar_grupo)
        despues = df["Grupo"].nunique()
        if despues < antes:
            utils.log(f"'Grupo': {antes} -> {despues} niveles al unificar "
                      "sinonimos singular/plural")

    return df


def excluir_ex_articulos(df: pd.DataFrame) -> pd.DataFrame:
    """Excluye los productos descontinuados (Status = 'Ex article').

    La comparacion se hace sobre el texto normalizado porque la fuente trae
    'Ex artcile' (con typo) y 'ex article' (minuscula) como valores distintos.
    """
    status_normalizado = df["Status"].map(utils.normalizar_texto)
    es_ex = status_normalizado.isin(cfg.STATUS_EXCLUIR)

    if es_ex.any():
        detalle = df.loc[es_ex, "Status"].value_counts().to_dict()
        utils.log(f"Excluyendo {int(es_ex.sum())} registros 'Ex article' -> {detalle}")
    else:
        utils.log("No se encontraron registros 'Ex article'")

    return df.loc[~es_ex].copy()


# ---------------------------------------------------------------------------
# VALIDACIONES
# ---------------------------------------------------------------------------


def validar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Valida unicidad de EAN y Material dentro de cada canal.

    El EAN tiene un duplicado real en la fuente, por lo que solo se reporta.
    El Material si debe ser unico: si hubiera duplicados, el modelo contaria
    dos veces el mismo SKU, asi que se conserva el registro con mas ventas.
    """
    utils.titulo("VALIDACION DE DUPLICADOS")
    meses = [m for m in cfg.MESES if m in df.columns]

    # --- EAN duplicado dentro del mismo canal (solo se reporta) -------------
    dup_ean = df.duplicated(subset=["EAN/UPC-Code", "Canal"], keep=False)
    if dup_ean.any():
        n_codigos = df.loc[dup_ean, "EAN/UPC-Code"].nunique()
        utils.log(
            f"{int(dup_ean.sum())} registros comparten EAN dentro de su canal "
            f"({n_codigos} codigo(s) EAN afectado(s)).",
            nivel="ALERTA",
        )
        columnas_detalle = ["Canal", "EAN/UPC-Code", "Material", "Marca", "Descripcion"]
        print(df.loc[dup_ean, columnas_detalle].to_string(index=False))
        utils.log(
            "Se conservan ambos: son materiales distintos con EAN mal cargado en "
            "la fuente. La clave del modelo es Material + Canal.",
        )
    else:
        utils.log("EAN/UPC-Code sin duplicados dentro de cada canal")

    # --- Material duplicado dentro del mismo canal (si se corrige) ----------
    dup_material = df.duplicated(subset=cfg.CLAVE_REGISTRO, keep=False)
    if dup_material.any():
        utils.log(
            f"{int(dup_material.sum())} registros con Material repetido en el "
            "mismo canal: se conserva el de mayor venta total.",
            nivel="ALERTA",
        )
        df = df.assign(_total=df[meses].sum(axis=1))
        df = (
            df.sort_values("_total", ascending=False)
              .drop_duplicates(subset=cfg.CLAVE_REGISTRO, keep="first")
              .drop(columns="_total")
        )
    else:
        utils.log("Material unico dentro de cada canal (clave valida)")

    return df


def validar_consistencia(df: pd.DataFrame) -> None:
    """Chequeos de sanidad sobre el maestro ya construido."""
    utils.titulo("VALIDACION DE CONSISTENCIA")
    meses = [m for m in cfg.MESES if m in df.columns]

    utils.log(f"Registros producto x canal: {len(df):,}")
    utils.log(f"Productos unicos (Material): {df['Material'].nunique():,}")
    utils.log(f"Canales: {sorted(df['Canal'].unique())}")
    utils.log(f"Marcas: {df['Marca'].value_counts().to_dict()}")
    utils.log(f"Status conservados: {df['Status'].value_counts().to_dict()}")

    total = df[meses].to_numpy().sum()
    utils.log(f"Unidades totales en el historico: {utils.formatear_miles(total)}")

    # Productos sin una sola unidad en 30 meses: no entran al clustering
    # principal, se etiquetan como 'Sin Movimiento' mas adelante.
    sin_movimiento = int((df[meses].sum(axis=1) == 0).sum())
    utils.log(
        f"Registros sin ninguna venta en 30 meses: {sin_movimiento} "
        f"({utils.porcentaje(sin_movimiento, len(df))}%)"
    )

    if df["PVP"].le(0).any():
        utils.log(f"{int(df['PVP'].le(0).sum())} registros con PVP <= 0", nivel="ALERTA")

    # Se avisa si alguna categorica es constante: no aporta al clustering y el
    # pipeline la descartara automaticamente al codificar.
    for columna in cfg.FEATURES_CATEGORICAS:
        if columna in df.columns and df[columna].nunique() == 1:
            utils.log(
                f"'{columna}' es constante ('{df[columna].iloc[0]}'): no aporta "
                "informacion al modelo y sera descartada al codificar.",
                nivel="ALERTA",
            )


# ---------------------------------------------------------------------------
# PIPELINE DE INGESTA
# ---------------------------------------------------------------------------


def construir_maestro(ruta_excel=None) -> pd.DataFrame:
    """Ejecuta la ingesta completa y devuelve el DataFrame maestro."""
    utils.titulo("PASO 1 - INGESTA Y UNIFICACION")

    hojas = leer_todas_las_hojas(ruta_excel)

    # Unificacion: al concatenar, 'Canal' distingue los registros del mismo
    # producto en distintos puntos de venta.
    maestro = pd.concat(hojas.values(), ignore_index=True)
    utils.log(f"Maestro unificado: {len(maestro):,} filas")

    meses = verificar_meses(maestro)
    maestro = limpiar_tipos(maestro, meses)
    maestro = excluir_ex_articulos(maestro)
    maestro = validar_duplicados(maestro)

    # Orden final de columnas: atributos, canal y luego los meses cronologicos.
    atributos = [c for c in cfg.COLUMNAS_ATRIBUTO if c in maestro.columns]
    maestro = maestro[atributos + ["Canal"] + meses].reset_index(drop=True)

    validar_consistencia(maestro)
    return maestro


def guardar_maestro(df: pd.DataFrame) -> str:
    """Guarda el maestro en data/processed/maestro.parquet."""
    cfg.crear_directorios()
    df.to_parquet(cfg.ARCHIVO_MAESTRO, index=False)
    utils.log(f"Maestro guardado en {cfg.ARCHIVO_MAESTRO}")
    return str(cfg.ARCHIVO_MAESTRO)


def cargar_maestro() -> pd.DataFrame:
    """Lee el maestro ya procesado; lo construye si aun no existe."""
    if not cfg.ARCHIVO_MAESTRO.exists():
        utils.log("maestro.parquet no existe todavia: ejecutando ingesta ...")
        return ejecutar()
    maestro = pd.read_parquet(cfg.ARCHIVO_MAESTRO)
    cfg.actualizar_calendario(detectar_meses(maestro))
    return maestro


def ejecutar(ruta_excel=None) -> pd.DataFrame:
    """Punto de entrada del paso 1: construye, valida y guarda el maestro."""
    maestro = construir_maestro(ruta_excel)
    guardar_maestro(maestro)
    return maestro


if __name__ == "__main__":
    ejecutar()
