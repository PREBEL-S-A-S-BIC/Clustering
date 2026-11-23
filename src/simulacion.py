"""Escenarios mensuales sobre el sugerido de lanzamientos."""

import numpy as np
import pandas as pd

from . import lanzamientos


def construir_plan(resultados, modelo, horizonte, anio):
    """Ubica la curva del modelo en los meses calendario de cada lanzamiento."""
    if horizonte not in (6, 12):
        raise ValueError("El horizonte debe ser de 6 o 12 meses.")
    columna_total = "unidades_sugeridas_anual" if horizonte == 12 else "unidades_sugeridas"
    if columna_total not in resultados:
        raise ValueError("El modelo no tiene un sugerido para ese horizonte.")
    factores = modelo.reparto_mensual_anual if horizonte == 12 else modelo.reparto_mensual
    factores = np.asarray(
        factores if factores is not None else np.full(horizonte, 1 / horizonte),
        dtype=float,
    )
    if (
        len(factores) != horizonte
        or not np.isfinite(factores).all()
        or (factores < 0).any()
        or not np.isfinite(factores.sum())
        or factores.sum() <= 0
    ):
        raise ValueError("La curva mensual del modelo es inválida.")
    factores = factores / factores.sum()
    metadatos = [c for c in ["Material", "Descripcion", "Marca", "Canal",
                            "Subcategoria", "Grupo", "Tipo", "Tono", "PVP"]
                 if c in resultados]
    filas = []
    for registro, (_, producto) in enumerate(resultados.iterrows(), start=1):
        mes = int(producto.get("mes_lanzamiento", 7))
        if mes not in range(1, 13):
            raise ValueError("El mes de lanzamiento debe estar entre 1 y 12.")
        total = float(producto[columna_total])
        if not np.isfinite(total) or total < 0:
            raise ValueError("El sugerido debe ser un número no negativo.")
        base = lanzamientos._repartir_entero(int(round(total)), factores)
        periodos = pd.period_range(f"{int(anio):04d}-{mes:02d}", periods=horizonte, freq="M")
        for posicion, periodo in enumerate(periodos):
            filas.append({
                "Registro": registro,
                **{c: producto[c] for c in metadatos},
                "Mes": str(periodo),
                "Mes de vida": posicion + 1,
                "Base del modelo": int(base[posicion]),
                "Factor base": float(factores[posicion]),
            })
    return pd.DataFrame(filas)


def aplicar_ajustes(plan, ajustes):
    """Redistribuye el total por pesos y luego aplica el cambio comercial."""
    columnas = ["Mes", "Peso estacional", "Ajuste comercial (%)", "Motivo"]
    if ajustes["Mes"].duplicated().any():
        raise ValueError("Cada mes debe tener una sola configuración.")
    resultado = plan.merge(ajustes[columnas], on="Mes", how="left", validate="many_to_one")
    for columna in ["Peso estacional", "Ajuste comercial (%)"]:
        resultado[columna] = pd.to_numeric(resultado[columna], errors="coerce")
        if not np.isfinite(resultado[columna]).all():
            raise ValueError(f"Completa todos los valores de {columna}.")
    if (resultado["Peso estacional"] < 0).any():
        raise ValueError("Los pesos estacionales deben ser mayores o iguales a cero.")
    if (resultado["Ajuste comercial (%)"] < -100).any():
        raise ValueError("La reducción comercial máxima es del 100 %.")
    resultado["Motivo"] = resultado["Motivo"].fillna("")
    resultado["Con estacionalidad"] = 0
    resultado["Unidades ajustadas"] = 0
    for _, grupo in resultado.groupby("Registro", sort=False):
        total = int(grupo["Base del modelo"].sum())
        pesos = grupo["Factor base"].to_numpy() * grupo["Peso estacional"].to_numpy()
        if total > 0 and pesos.sum() <= 0:
            raise ValueError("Asigna al menos un peso positivo en cada período del producto.")
        if not np.isfinite(pesos).all() or not np.isfinite(pesos.sum()):
            raise ValueError("Los pesos estacionales son demasiado grandes.")
        estacional = lanzamientos._repartir_entero(
            total, pesos / pesos.sum() if pesos.sum() > 0 else np.zeros(len(pesos))
        )
        valores = estacional * (1 + grupo["Ajuste comercial (%)"].to_numpy() / 100)
        if not np.isfinite(valores).all() or valores.sum() >= np.iinfo(np.int64).max:
            raise ValueError("El ajuste comercial produce un volumen demasiado grande.")
        total_ajustado = int(np.floor(valores.sum() + 0.5))
        ajustado = lanzamientos._repartir_entero(
            total_ajustado,
            valores / valores.sum() if valores.sum() > 0 else np.zeros(len(valores)),
        )
        resultado.loc[grupo.index, "Con estacionalidad"] = estacional
        resultado.loc[grupo.index, "Unidades ajustadas"] = ajustado
    resultado["Diferencia"] = resultado["Unidades ajustadas"] - resultado["Base del modelo"]
    return resultado
