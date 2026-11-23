"""Convierte ventas POS de SAP al maestro que consume el pipeline."""

import pandas as pd

from . import config as cfg
from . import ingesta, utils


def construir_maestro(ventas, atributos, inicio, fin, canales):
    """Conserva el catálogo y reconstruye las ventas del período solicitado."""
    columnas = cfg.COLUMNAS_ATRIBUTO + ["Canal"]
    faltantes = set(columnas) - set(atributos.columns)
    if faltantes:
        raise ValueError(f"Faltan atributos en el maestro: {sorted(faltantes)}")
    catalogo = atributos[columnas].copy()
    catalogo["Material"] = catalogo["Material"].astype(str).str.strip()
    if catalogo.duplicated(["Material", "Canal"]).any():
        raise ValueError("El maestro de atributos tiene material y canal repetidos.")
    datos = ventas.copy()
    datos["Material"] = datos["Material-CHAVL"].astype(str).str.strip()
    datos["Canal"] = datos["Nit-CHAVL"].astype(str).str.strip().map(canales)
    if datos["Canal"].isna().any():
        raise ValueError("Hay NIT sin correspondencia con un canal.")
    marca = datos["Marca Jerarquia de Productos-CAPTION"].map(utils.normalizar_texto)
    datos = datos[marca.isin(["essence", "catrice"])].copy()
    fuera_catalogo = ~datos["Material"].isin(catalogo["Material"])
    utils.log(
        f"Materiales SAP fuera del Excel: "
        f"{datos.loc[fuera_catalogo, 'Material'].nunique()} "
        "(se incorporan al agregarlos al catálogo)."
    )
    datos = datos.loc[~fuera_catalogo].copy()
    if datos.empty:
        raise ValueError("SAP no devolvió ventas para los materiales del catálogo en el período.")
    fechas = pd.to_datetime(datos["fecha"])
    if not fechas.between(pd.Timestamp(inicio), pd.Timestamp(fin) + pd.offsets.MonthEnd(0)).all():
        raise ValueError("Las ventas incluyen fechas fuera del período solicitado.")
    claves = datos[["Material", "Canal"]].drop_duplicates()
    nuevas = claves.merge(catalogo[["Material", "Canal"]], how="left",
                          indicator=True, on=["Material", "Canal"])
    nuevas = nuevas[nuevas["_merge"] == "left_only"][["Material", "Canal"]]
    if not nuevas.empty:
        # Un material conocido puede incorporarse a otro canal si sus atributos
        # son idénticos en todos los canales del maestro.
        comunes = catalogo[cfg.COLUMNAS_ATRIBUTO].drop_duplicates()
        ambiguos = comunes.loc[comunes.duplicated("Material", keep=False), "Material"]
        faltan = nuevas.loc[
            ~nuevas["Material"].isin(comunes["Material"])
            | nuevas["Material"].isin(ambiguos), "Material"
        ].unique()
        if len(faltan):
            raise ValueError(
                "Completa los atributos de estos materiales antes de entrenar: "
                + ", ".join(sorted(faltan)[:20])
                + f" ({len(faltan)} materiales en total)."
            )
        altas = nuevas.merge(comunes, on="Material", validate="many_to_one")
        catalogo = pd.concat([catalogo, altas[columnas]], ignore_index=True)

    datos["mes"] = fechas.loc[datos.index].dt.to_period("M")
    mensual = datos.groupby(["Material", "Canal", "mes"])["Ventas POS (Und)"].sum()
    matriz = mensual.unstack("mes", fill_value=0)
    periodos = pd.period_range(inicio, fin, freq="M")
    matriz = matriz.reindex(columns=periodos, fill_value=0)
    meses = [f"{ingesta.ABREV_MES[p.month]}-{str(p.year)[-2:]}" for p in periodos]
    matriz.columns = meses
    maestro = catalogo.merge(matriz.reset_index(), on=["Material", "Canal"],
                             how="left", validate="one_to_one")
    maestro[meses] = maestro[meses].fillna(0)
    # El EAN SAP se conserva como texto y se valida por material.
    ean = datos[["Material", "ean"]].drop_duplicates()
    ean = ean[ean["ean"].notna() & ~ean["ean"].isin(["", "Sin EAN"])]
    if ean.duplicated("Material").any():
        raise ValueError("SAP devolvió EAN contradictorios para un material.")
    maestro["EAN/UPC-Code"] = (
        maestro["Material"].map(ean.set_index("Material")["ean"])
        .fillna(maestro["EAN/UPC-Code"])
    )
    cfg.actualizar_calendario(meses)
    maestro = ingesta.limpiar_tipos(maestro, meses)
    maestro = ingesta.excluir_ex_articulos(maestro)
    ingesta.validar_consistencia(maestro)
    return maestro.reset_index(drop=True)
