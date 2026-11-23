# -*- coding: utf-8 -*-
"""
run_pipeline.py - Ejecuta el proyecto completo de punta a punta.

Corre los 5 pasos en orden y deja todos los artefactos listos:
    1. Ingesta y unificacion         -> data/processed/maestro.parquet
    2. Feature engineering           -> data/processed/features.parquet
    3. Clustering                    -> data/output/clusters.parquet
    4. Interpretacion y proyeccion   -> data/output/perfiles_cluster.csv
    5. Figuras y Excel final         -> outputs/figures/, Cosnova_Clusters_Final.xlsx

Uso:
    python run_pipeline.py
    python run_pipeline.py --sin-figuras     (mas rapido: omite t-SNE y graficas)
"""

from __future__ import annotations

import argparse
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Permite ejecutar el script desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (clustering, eda, evaluation, feature_engineering, ingesta,
                 lanzamientos, utils)
from src import config as cfg


def generar_figuras_eda(maestro, features) -> None:
    """Genera las 7 figuras del analisis exploratorio (figuras 01 a 04d)."""
    import matplotlib.pyplot as plt

    utils.aplicar_estilo_graficas()
    utils.titulo("GENERANDO FIGURAS DEL EDA")

    eda.graficar_distribucion_por_dimension(maestro)
    eda.graficar_evolucion_mensual(maestro)
    eda.graficar_heatmap_productos(maestro, top=50)
    eda.graficar_cobertura(features)
    eda.graficar_boxplots(features)
    eda.graficar_tendencias(features)
    eda.graficar_correlaciones(features)

    plt.close("all")


def generar_figuras_clusters(df, perfiles, resultado) -> None:
    """Genera las 5 figuras del modelo (figuras 05 a 09)."""
    import matplotlib.pyplot as plt

    utils.aplicar_estilo_graficas()
    utils.titulo("GENERANDO FIGURAS DE CLUSTERS")

    evaluation.graficar_pca(df)
    evaluation.graficar_perfil_clusters(df)
    evaluation.graficar_metricas_por_cluster(perfiles)

    # Dendrograma del modelo jerarquico
    matriz_enlace, _ = clustering.calcular_linkage(resultado.matriz_pca)
    fig, ax = plt.subplots(figsize=(14, 6))
    clustering.graficar_dendrograma(matriz_enlace, resultado.k_optimo, ax=ax)
    utils.guardar_figura(fig, "09_dendrograma")

    # t-SNE al final porque es la figura mas costosa de calcular
    evaluation.graficar_tsne(df, resultado.matriz_pca)

    plt.close("all")


def generar_figuras_lanzamientos(modelo_lanz) -> None:
    """Genera las 2 figuras del modelo de lanzamientos (figuras 10 y 11)."""
    import matplotlib.pyplot as plt

    utils.aplicar_estilo_graficas()
    utils.titulo("GENERANDO FIGURAS DE LANZAMIENTOS")

    lanzamientos.graficar_importancia(modelo_lanz)
    lanzamientos.graficar_distribucion_target(modelo_lanz)

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de clustering Cosnova")
    parser.add_argument("--sin-figuras", action="store_true",
                        help="Omite la generacion de graficas (ejecucion mas rapida)")
    parser.add_argument("--actualizar-sap", action="store_true",
                        help="Descarga POS y reconstruye el maestro antes de entrenar.")
    parser.add_argument("--desde-maestro", action="store_true",
                        help="Entrena desde el maestro procesado sin repetir la descarga.")
    parser.add_argument("--credenciales", type=Path,
                        help="JSON de conexión SAP.")
    parser.add_argument("--python-sap", type=Path, default=Path(sys.executable),
                        help="Python con pyrfc y pyarrow para ejecutar la descarga.")
    parser.add_argument("--inicio", default="2024-01", help="Primer mes SAP, AAAA-MM.")
    hoy = datetime.now(timezone(timedelta(hours=-5))).date()
    ultimo_cerrado = hoy.replace(day=1) - timedelta(days=1)
    parser.add_argument("--fin", default=ultimo_cerrado.strftime("%Y-%m"),
                        help="Último mes SAP, por defecto el último mes cerrado.")
    parser.add_argument("--atributos", type=Path, default=cfg.ARCHIVO_EXCEL,
                        help="Excel por canal o Parquet con los atributos comerciales.")
    parser.add_argument("--solo-features", action="store_true",
                        help="Actualiza maestro y features sin entrenar.")
    args = parser.parse_args()
    if args.actualizar_sap and args.desde_maestro:
        parser.error("Elige --actualizar-sap o --desde-maestro.")
    if args.actualizar_sap and args.credenciales is None:
        parser.error("--actualizar-sap requiere --credenciales.")

    inicio = time.time()
    cfg.crear_directorios()

    # Deja claro con que Python se esta ejecutando. Si es el del sistema en vez
    # del entorno virtual, avisa: hoy funciona igual, pero las versiones pueden
    # diverger y el problema seria invisible.
    utils.titulo("ENTORNO DE EJECUCION")
    utils.log_interprete()

    if not args.actualizar_sap and not args.desde_maestro and not cfg.ARCHIVO_EXCEL.exists():
        utils.log(f"No se encuentra el archivo fuente: {cfg.ARCHIVO_EXCEL}", nivel="ERROR")
        sys.exit(1)

    # Paso 1-2: datos
    if args.actualizar_sap:
        import pandas as pd
        from src import sap_ingesta
        # Leer antes de publicar resultados, incluso si se usa maestro.parquet.
        if args.atributos.suffix.lower() == ".xlsx":
            atributos = pd.concat(
                ingesta.leer_todas_las_hojas(args.atributos).values(), ignore_index=True
            )
        else:
            atributos = pd.read_parquet(args.atributos)
        configuracion = json.loads((cfg.RAIZ / "sap_zpos_config.json").read_text(encoding="utf-8"))
        comando = [
            str(args.python_sap), str(cfg.RAIZ / "extraer_zpos.py"),
            "--inicio", args.inicio, "--fin", args.fin,
            "--credenciales", str(args.credenciales.resolve()),
        ]
        for nit in configuracion["canales"]:
            comando.extend(["--nit", nit])
        subprocess.run(comando, check=True)
        ventas = pd.read_parquet(cfg.DIR_RAW / "ventas_pos_sap.parquet")
        maestro = sap_ingesta.construir_maestro(
            ventas, atributos, args.inicio, args.fin, configuracion["canales"]
        )
        # Construir ambas bases antes de reemplazar los resultados anteriores.
        features = feature_engineering.construir_features(maestro)
        from extraer_zpos import guardar_atomico
        guardar_atomico(maestro, cfg.ARCHIVO_MAESTRO)
        guardar_atomico(features, cfg.ARCHIVO_FEATURES)
    elif args.desde_maestro:
        maestro = ingesta.cargar_maestro()
        features = feature_engineering.ejecutar(maestro)
    else:
        maestro = ingesta.ejecutar()
        features = feature_engineering.ejecutar(maestro)
    if args.solo_features:
        utils.log("Maestro y features actualizados.")
        return

    # Figuras del EDA: se generan aqui porque solo dependen de los pasos 1 y 2
    if not args.sin_figuras:
        generar_figuras_eda(maestro, features)

    # Paso 3: modelo
    resultado = clustering.ejecutar(features)

    # Paso 4: interpretacion
    df_final, perfiles = evaluation.ejecutar(resultado)

    # Paso 5: sugerido para productos sin historial (modelo de atributos).
    # Es un modelo aparte: el clustering proyecta desde la historia de venta, y
    # estos productos no tienen historia. Predice desde sus atributos.
    modelo_lanz, sugeridos = lanzamientos.ejecutar(maestro)

    # Paso 6: figuras del modelo y entregable
    if not args.sin_figuras:
        generar_figuras_clusters(df_final, perfiles, resultado)
        generar_figuras_lanzamientos(modelo_lanz)

    excel_salida = cfg.ARCHIVO_EXCEL_FINAL
    try:
        utils.exportar_excel_final(df_final, perfiles, sugeridos)
    except PermissionError:
        marca_tiempo = datetime.now(timezone(timedelta(hours=-5))).strftime("%Y%m%d_%H%M%S")
        excel_salida = cfg.DIR_OUTPUTS / f"Cosnova_Clusters_{marca_tiempo}.xlsx"
        utils.log(f"Excel principal bloqueado. Exportando a {excel_salida}", nivel="ALERTA")
        utils.exportar_excel_final(
            df_final, perfiles, sugeridos, ruta_salida=excel_salida
        )

    utils.titulo("PIPELINE COMPLETADO")
    utils.log(f"Modelo de clusters: {resultado.modelo_ganador} (k={resultado.k_optimo})")
    utils.log(f"Registros clasificados: {len(df_final):,}")
    utils.log(f"Lanzamientos pendientes con sugerido: {len(sugeridos):,} "
              f"({utils.formatear_miles(sugeridos['unidades_sugeridas_lanzamiento'].sum())} und)")
    utils.log(f"Excel final: {excel_salida}")
    utils.log(f"Tiempo total: {time.time() - inicio:.1f} segundos")


if __name__ == "__main__":
    main()
