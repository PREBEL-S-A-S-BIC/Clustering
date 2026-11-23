# -*- coding: utf-8 -*-
"""
Configuracion central del proyecto de clustering Cosnova.

Aqui viven todas las rutas, constantes y parametros del modelo para que
ningun otro modulo tenga rutas "hardcodeadas".
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS DEL PROYECTO
# ---------------------------------------------------------------------------
# config.py vive en src/, por lo tanto la raiz del proyecto es el padre de src/
RAIZ = Path(__file__).resolve().parents[1]

DIR_DATA = RAIZ / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_OUTPUT_DATA = DIR_DATA / "output"

DIR_OUTPUTS = RAIZ / "outputs"
DIR_FIGURAS = DIR_OUTPUTS / "figures"

DIR_NOTEBOOKS = RAIZ / "notebooks"
DIR_DASHBOARD = RAIZ / "dashboard"

# Archivo fuente original (Excel entregado por el negocio)
ARCHIVO_EXCEL = DIR_RAW / "Metodo Planeación Cosnova News 2026 editado.xlsx"

# Artefactos intermedios (formato parquet: rapido y conserva tipos de datos)
ARCHIVO_MAESTRO = DIR_PROCESSED / "maestro.parquet"
ARCHIVO_FEATURES = DIR_PROCESSED / "features.parquet"
ARCHIVO_CLUSTERS = DIR_OUTPUT_DATA / "clusters.parquet"
ARCHIVO_METRICAS = DIR_OUTPUT_DATA / "metricas_modelos.csv"

# Entregable final para el negocio
ARCHIVO_EXCEL_FINAL = DIR_OUTPUTS / "Cosnova_Clusters_Final.xlsx"


def crear_directorios() -> None:
    """Crea todas las carpetas del proyecto si aun no existen (idempotente)."""
    for carpeta in (
        DIR_RAW,
        DIR_PROCESSED,
        DIR_OUTPUT_DATA,
        DIR_FIGURAS,
        DIR_OUTPUTS,
        DIR_NOTEBOOKS,
        DIR_DASHBOARD,
    ):
        carpeta.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ESTRUCTURA DEL ARCHIVO FUENTE
# ---------------------------------------------------------------------------
# El Excel tiene 6 hojas, pero solo estas 3 son hojas de canal con historico
# mensual. Las hojas 'BD', 'Ventas' y 'TD POS' son insumos previos del negocio
# y NO se usan en este modelo.
HOJAS_CANAL = ["Exito", "Farmatodo", "Prosalon"]

# Columnas descriptivas (todo lo que no es un mes de venta).
# OJO: en el Excel original 'Material' y 'Descripcion' traen un espacio final
# en el encabezado; ingesta.py normaliza los nombres antes de usarlos.
COLUMNAS_ATRIBUTO = [
    "Marca",
    "EAN/UPC-Code",
    "Status",
    "Tipo Portafolio",
    "Material",
    "Descripcion",
    "Categoria",
    "Subcategoria",
    "Grupo",
    "Tono",
    "Tipo",
    "Caracteristica",
    "PVP",
]

# Clave unica de un registro del modelo.
# Se usa Material (codigo SKU interno) y NO el EAN, porque el EAN tiene un
# duplicado real en la fuente: el codigo 4059729583222 aparece tanto en un
# material Catrice como en uno Essence. Material si es unico por hoja.
CLAVE_PRODUCTO = "Material"
CLAVE_REGISTRO = ["Material", "Canal"]

# ---------------------------------------------------------------------------
# UNIFICACION DE SINONIMOS EN 'Grupo'
# ---------------------------------------------------------------------------
# La fuente trae el mismo grupo escrito en singular y en plural, como si fueran
# categorias distintas. Los casos verificados y su peso en registros:
#
#   Lapiz de ojos (108)  /  Lapices de ojos (21)
#   Pestanina (126)      /  Pestaninas (30)
#   Corrector (75)       /  Correctores (15)
#   Base (63)            /  Bases (15)
#   Polvo (48)           /  Polvos (18)
#   Lapiz de cejas (12)  /  Lapices de cejas (21)
#   Brocha (3)           /  Brochas (3)
#
# Dejarlos separados fragmenta las cohortes de productos comparables sin ninguna
# razon de negocio: un lanzamiento de 'Pestaninas' no encontraria los 126
# comparables de 'Pestanina'. Las llaves van sin tildes porque la limpieza de
# categorias las elimina antes de aplicar este mapa.
#
# No se unifican 'Tinta', 'Tinta labial' y 'Tinta de labios': podrian ser tinta
# de labios o de unas y no hay forma de saberlo desde el dato.
SINONIMOS_GRUPO = {
    "Pestaninas": "Pestanina",
    "Correctores": "Corrector",
    "Polvos": "Polvo",
    "Brochas": "Brocha",
    "Bases": "Base",
    "Lapices de ojos": "Lapiz de ojos",
    "Lapices de cejas": "Lapiz de cejas",
    "Lapices de labios": "Lapiz de labios",
}

# Valores de Status que representan articulos descontinuados y se excluyen.
# En la fuente vienen con inconsistencias de escritura: 'Ex artcile' (typo) y
# 'ex article' (minuscula). La comparacion se hace normalizada.
STATUS_EXCLUIR = ["ex article", "ex artcile", "exarticle"]

# ---------------------------------------------------------------------------
# CALENDARIO: 30 MESES DE HISTORIA (enero 2024 - junio 2026)
# ---------------------------------------------------------------------------
MESES = [
    "ene-24", "feb-24", "mar-24", "abr-24", "may-24", "jun-24",
    "jul-24", "ago-24", "sep-24", "oct-24", "nov-24", "dic-24",
    "ene-25", "feb-25", "mar-25", "abr-25", "may-25", "jun-25",
    "jul-25", "ago-25", "sep-25", "oct-25", "nov-25", "dic-25",
    "ene-26", "feb-26", "mar-26", "abr-26", "may-26", "jun-26",
]

# Agrupacion semestral usada en todo el proyecto.
# S1_2026 solo tiene 6 meses cerrados; S2_2026 es justamente lo que se planea.
SEMESTRES = {
    "S1_2024": ["ene-24", "feb-24", "mar-24", "abr-24", "may-24", "jun-24"],
    "S2_2024": ["jul-24", "ago-24", "sep-24", "oct-24", "nov-24", "dic-24"],
    "S1_2025": ["ene-25", "feb-25", "mar-25", "abr-25", "may-25", "jun-25"],
    "S2_2025": ["jul-25", "ago-25", "sep-25", "oct-25", "nov-25", "dic-25"],
    "S1_2026": ["ene-26", "feb-26", "mar-26", "abr-26", "may-26", "jun-26"],
}

# Semestre que se va a planear con el modelo.
SEMESTRE_OBJETIVO = "S2_2026"

# Semestres historicos del mismo tipo que el objetivo (segundos semestres).
# Sirven de base para la proyeccion: S2_2026 se estima mirando S2 pasados.
SEMESTRES_HOMOLOGOS = ["S2_2024", "S2_2025"]

# Semestres completos (12 meses por año) para calcular estacionalidad limpia.
SEMESTRES_S1_COMPLETOS = ["S1_2024", "S1_2025"]
SEMESTRES_S2_COMPLETOS = ["S2_2024", "S2_2025"]

TOTAL_MESES = len(MESES)  # 30

# Los valores de arriba son el punto de partida, NO una definicion fija. Cuando
# el negocio agrega al Excel la venta POS de meses nuevos, `ingesta` detecta el
# calendario real y llama a `actualizar_calendario` para reemplazarlos.
#
# Sin eso, agregar jul-26 al archivo no serviria de nada: todos los modulos
# filtran con `[m for m in cfg.MESES if m in df.columns]`, asi que un mes que no
# estuviera en esta lista se descartaria en silencio y el modelo seguiria
# calculando sobre los 30 meses viejos sin avisar.

_ORDEN_MES = {m: i + 1 for i, m in enumerate(
    ["ene", "feb", "mar", "abr", "may", "jun",
     "jul", "ago", "sep", "oct", "nov", "dic"])}


def ordenar_meses(meses) -> list:
    """Ordena etiquetas 'ene-24' cronologicamente (anio y luego mes)."""
    def clave(etiqueta):
        abrev, anio = str(etiqueta).split("-")
        return (int(anio), _ORDEN_MES.get(abrev, 0))
    return sorted(meses, key=clave)


def construir_semestres(meses: list) -> dict:
    """Agrupa una lista de meses en semestres S1_YYYY / S2_YYYY."""
    grupos: dict[str, list] = {}
    for etiqueta in ordenar_meses(meses):
        abrev, anio = str(etiqueta).split("-")
        numero = _ORDEN_MES.get(abrev, 0)
        semestre = f"S{1 if numero <= 6 else 2}_20{anio}"
        grupos.setdefault(semestre, []).append(etiqueta)
    return grupos


def semestres_completos() -> list:
    """Semestres con sus 6 meses, en orden cronologico."""
    return [s for s, m in SEMESTRES.items() if len(m) == 6]


def semestre_base() -> str:
    """El semestre desde el que se proyecta: el ultimo COMPLETO antes del objetivo.

    Se exigen dos cosas y ambas importan:

    1. **Que no sea el propio objetivo.** Cuando el archivo ya trae meses sueltos
       del semestre a planear (por ejemplo jul y ago de un S2_2026 en curso), el
       "ultimo semestre con datos" ES el objetivo, y proyectarlo desde si mismo
       seria circular.
    2. **Que este completo.** Un semestre a medias tiene menos meses acumulados y
       haria parecer que el producto cayo, cuando solo falta el resto del periodo.
    """
    completos = [s for s in semestres_completos() if s != SEMESTRE_OBJETIVO]
    if completos:
        return completos[-1]
    # Sin ningun semestre cerrado se cae al ultimo con datos que no sea el objetivo
    disponibles = [s for s in SEMESTRES if s != SEMESTRE_OBJETIVO]
    return disponibles[-1] if disponibles else ""


def par_semestral_reciente() -> tuple:
    """Ultimo par (S1, S2) completo del mismo anio.

    Sirve para calibrar el factor de paso de un semestre al otro: se prefiere el
    anio mas reciente que tenga los dos semestres cerrados.
    """
    completos = semestres_completos()
    por_anio: dict[str, dict] = {}
    for s in completos:
        mitad, anio = s.split("_")
        por_anio.setdefault(anio, {})[mitad] = s
    for anio in sorted(por_anio, reverse=True):
        if "S1" in por_anio[anio] and "S2" in por_anio[anio]:
            return por_anio[anio]["S1"], por_anio[anio]["S2"]
    return ("", "")


def _siguiente_semestre(semestre: str) -> str:
    """Devuelve el semestre que sigue: S1_2026 -> S2_2026 -> S1_2027."""
    mitad, anio = semestre.split("_")
    return f"S2_{anio}" if mitad == "S1" else f"S1_{int(anio) + 1}"


def actualizar_calendario(meses_detectados: list) -> dict:
    """Reemplaza el calendario del proyecto por el que trae el Excel.

    Recalcula los meses, los semestres, cuales estan completos, cual es el
    semestre a planear y sus homologos historicos. Asi, agregar meses al archivo
    fuente basta para que todo el modelo los tome en cuenta.

    Returns:
        Un resumen de los cambios, para poder informarlos en pantalla.
    """
    global MESES, SEMESTRES, TOTAL_MESES, SEMESTRE_OBJETIVO
    global SEMESTRES_HOMOLOGOS, SEMESTRES_S1_COMPLETOS, SEMESTRES_S2_COMPLETOS

    nuevos = ordenar_meses(meses_detectados)
    if not nuevos:
        return {"cambio": False, "motivo": "no se detecto ningun mes"}

    anteriores = list(MESES)
    semestres = construir_semestres(nuevos)

    # Solo los semestres con sus 6 meses sirven para comparar semestre contra
    # semestre; uno a medias distorsionaria la estacionalidad y la tendencia.
    completos = {s: m for s, m in semestres.items() if len(m) == 6}

    MESES = nuevos
    SEMESTRES = semestres
    TOTAL_MESES = len(nuevos)
    SEMESTRES_S1_COMPLETOS = [s for s in completos if s.startswith("S1")]
    SEMESTRES_S2_COMPLETOS = [s for s in completos if s.startswith("S2")]

    # El semestre a planear es el siguiente al ultimo con datos completos.
    ultimo_completo = list(completos)[-1] if completos else list(semestres)[-1]
    SEMESTRE_OBJETIVO = _siguiente_semestre(ultimo_completo)

    # Homologos: los semestres de la misma mitad del ano ya observados. Son la
    # base de comparacion para proyectar (un S2 se estima mirando otros S2).
    mitad = SEMESTRE_OBJETIVO.split("_")[0]
    SEMESTRES_HOMOLOGOS = [s for s in completos if s.startswith(mitad)]

    agregados = [m for m in nuevos if m not in anteriores]
    quitados = [m for m in anteriores if m not in nuevos]

    return {
        "cambio": bool(agregados or quitados),
        "meses_agregados": agregados,
        "meses_quitados": quitados,
        "total_meses": TOTAL_MESES,
        "primer_mes": nuevos[0],
        "ultimo_mes": nuevos[-1],
        "semestre_objetivo": SEMESTRE_OBJETIVO,
        "semestres_completos": list(completos),
    }

# ---------------------------------------------------------------------------
# PARAMETROS DEL MODELO
# ---------------------------------------------------------------------------
SEMILLA = 42  # reproducibilidad en KMeans, PCA y t-SNE

# Variables numericas de comportamiento que alimentan el clustering.
FEATURES_NUMERICAS = [
    "total_unidades",
    "promedio_mensual",
    "cobertura_meses",
    "tendencia",
    "estacionalidad",
    "variabilidad",
]

# Variables categoricas que se codifican con One-Hot para el modelo.
# NOTA: 'Categoria' se incluye por requerimiento, pero en esta fuente es
# constante ('Maquillaje'); el pipeline descarta automaticamente las columnas
# de varianza cero, asi que no aporta ruido. La jerarquia informativa real es
# Subcategoria -> Grupo.
FEATURES_CATEGORICAS = [
    "Categoria",
    "Subcategoria",
    "Marca",
    "Canal",
    "Tipo Portafolio",
]

# Busqueda del numero de clusters
K_MIN = 2
K_MAX = 12

# Varianza acumulada que debe conservar el PCA
VARIANZA_PCA = 0.90

# DBSCAN: se calibra eps automaticamente hasta que el ruido baje de este techo
MAX_RUIDO_DBSCAN = 0.10

# Etiqueta y cluster reservados para productos sin una sola unidad vendida
CLUSTER_SIN_MOVIMIENTO = -1
NOMBRE_SIN_MOVIMIENTO = "Sin Movimiento"

# Umbrales del semaforo de prioridad de compra
UMBRAL_COBERTURA_ALTA = 70.0   # %
UMBRAL_COBERTURA_BAJA = 40.0   # %
UMBRAL_TENDENCIA_NEGATIVA = -1.0  # unidades/mes: caida considerada fuerte

# Estilo unico para todas las figuras del proyecto
PALETA = "viridis"
DPI_FIGURAS = 120
