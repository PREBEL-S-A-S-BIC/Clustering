# -*- coding: utf-8 -*-
"""
Paquete `src` del proyecto de clustering de productos Cosnova (ESSENCE / CATRICE).

Modulos:
    config              -> rutas, constantes y parametros del modelo
    utils               -> utilidades comunes y exportacion del Excel final
    ingesta             -> lectura y unificacion de las 3 hojas de canal
    feature_engineering -> construccion de variables de comportamiento
    clustering          -> preprocesamiento, k optimo y entrenamiento de modelos
    evaluation          -> perfilamiento, nombres de cluster y prioridad de compra
    eda                 -> figuras del analisis exploratorio
    lanzamientos        -> sugerido de unidades para productos SIN historial

Hay dos modelos, no uno, porque son dos problemas distintos:

    clustering + evaluation -> proyectan desde la HISTORIA de venta.
                               Sirven para productos que ya venden.
    lanzamientos            -> predice desde los ATRIBUTOS del producto
                               (canal, marca, grupo, tipo, tono, PVP, mes).
                               Sirve para productos que aun no tienen historia.

Uso tipico (pipeline completo):

    from src import ingesta, feature_engineering, clustering, evaluation
    from src import lanzamientos, utils

    maestro   = ingesta.ejecutar()
    features  = feature_engineering.ejecutar()
    resultado = clustering.ejecutar()
    final, perfiles = evaluation.ejecutar()
    modelo, sugeridos = lanzamientos.ejecutar()
    utils.exportar_excel_final(final, perfiles, sugeridos)

Sugerido para un producto a lanzar:

    modelo = lanzamientos.entrenar()
    lanzamientos.sugerir_lanzamiento(
        modelo, canal="Farmatodo", subcategoria="Labios", grupo="Labial",
        tipo="Barra", tono="Rojo", pvp=23000, mes_lanzamiento=7)
"""

__version__ = "1.1.0"
__all__ = [
    "config",
    "utils",
    "ingesta",
    "feature_engineering",
    "clustering",
    "evaluation",
    "eda",
    "lanzamientos",
]
