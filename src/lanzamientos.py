# -*- coding: utf-8 -*-
"""
lanzamientos.py - Sugerido de unidades para productos SIN historial de venta.

Responde la pregunta: "voy a lanzar un labial tono Rojo, PVP 23.000, grupo Labial,
tipo Barra. Cuantas unidades pido para la primera compra?"

Es un problema DISTINTO al clustering. El clustering agrupa por historia de venta;
un producto a lanzar no tiene historia, asi que hay que predecir desde sus
atributos: canal, marca, subcategoria, grupo, tipo, tono, PVP y mes de lanzamiento.

--------------------------------------------------------------------------------
QUE PREDICE
--------------------------------------------------------------------------------
`und_primeros_6m`: unidades vendidas en los primeros 6 meses de vida del producto
en ese canal. Es exactamente lo que se pide en la primera compra semestral.

--------------------------------------------------------------------------------
COMO SE ARMA EL SET DE ENTRENAMIENTO (la decision mas delicada)
--------------------------------------------------------------------------------
De los 1.815 registros producto x canal solo sirven 916. El filtro importa:

    Nunca vendieron ................ 385  -> son los que hay que PREDECIR
    Ya vendian en ene-24 ........... 352  -> se EXCLUYEN (censura a la izquierda)
    Debut limpio + 6 meses obs. .... 916  -> ENTRENAMIENTO
    Debut hace menos de 6 meses .... 162  -> se excluyen (target incompleto)

**Por que se excluyen los censurados.** Un producto que ya vendia en enero de 2024
(el primer mes de la ventana) pudo haber lanzado en 2023 o en 2019: su debut real
es desconocido. Tomar sus primeros 6 meses observados como si fuera un lanzamiento
mediria un producto en regimen, no un arranque. Serian 352 etiquetas equivocadas
sobre 1.268, suficiente para sesgar el modelo hacia arriba de forma sistematica.

--------------------------------------------------------------------------------
DOS MOTORES QUE SE CRUZAN
--------------------------------------------------------------------------------
1. **Modelo cuantilico** (Gradient Boosting con loss="quantile"): entrega P25, P50
   y P75 directamente. Es la cifra oficial del sugerido.
2. **Cohorte de analogos**: busca productos historicos parecidos en cascada, de lo
   mas especifico a lo mas general, y muestra sus cifras reales. Sirve para que el
   comprador VEA de donde sale el numero y para detectar cuando el modelo se sale
   de lo que respalda la evidencia.

Si ambos difieren por mas del doble, el registro se marca para revision manual.

--------------------------------------------------------------------------------
VALIDACION (GroupKFold por Material, 5 folds)
--------------------------------------------------------------------------------
                                     MAE    MAE_log   dentro factor 2
    Mediana global                   392     1.246         32.4%
    Mediana por canal                312     0.916         44.2%
    Mediana canal + subcategoria     306     0.888         48.5%
    RandomForest                     228     0.627         63.9%   <- +31.5%

**Por que GroupKFold y no KFold normal.** El mismo Material aparece en los 3
canales con targets correlacionados. Con KFold simple, Exito puede caer en train y
Farmatodo en test: el modelo "acierta" por fuga de informacion y la metrica miente.
Agrupar por Material cierra esa fuga.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as cfg
from . import ingesta, utils

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
MESES_HORIZONTE = 6           # primeros 6 meses de vida = primera compra semestral

# Horizonte anual: los primeros 12 meses de rotacion. Se entrena como un modelo
# APARTE y no se extrapola del de 6 meses.
#
# Se compararon las dos alternativas con validacion cruzada agrupada por Material:
#     Modelo directo a 12 meses        MAE_log 0.649   (+30,3% sobre el baseline)
#     Extrapolar 6m x ratio del canal  MAE_log 0.659   (+29,4%)
# Quedaron practicamente empatados, y se eligio el directo por ser algo mejor y
# por no encadenar dos estimaciones (el error de la primera no se arrastra).
#
# El set de entrenamiento anual es mas chico (686 lanzamientos con 12 meses
# observables, contra 916 con 6), que es el costo de exigir mas historia.
MESES_HORIZONTE_ANUAL = 12

# Nombres de las columnas del desagregado mes a mes de rotacion.
#
# Se define explicitamente y no se filtra por el prefijo "mes_": la columna
# 'mes_lanzamiento' (el mes calendario en que sale el producto) tambien empieza
# asi, y colarla en el reparto mensual la sumaria como si fuera unidades.
COLUMNAS_MES = [f"mes_{k + 1:02d}" for k in range(MESES_HORIZONTE_ANUAL)]

# Umbral para considerar que el primer mes de un lanzamiento fue COMPLETO.
#
# El "mes 1" se define como el primer mes con venta, pero un producto que entra
# el dia 20 solo alcanza a vender diez dias: ese mes queda medido a medias y
# arrastra la curva hacia abajo. Los lanzamientos cuyo mes 1 llega al menos al
# 80% del mes 2 son los que muy probablemente entraron a comienzos de mes, y son
# los unicos que sirven para estimar cuanto vende un primer mes completo.
UMBRAL_PRIMER_MES_COMPLETO = 0.8

# Minimo de casos con primer mes completo para aplicar la correccion. Por debajo
# de esto la estimacion seria mas ruido que senal y se deja la curva sin tocar.
MIN_CASOS_PRIMER_MES = 30

CUANTILES = [0.25, 0.50, 0.75]
CUANTIL_SUGERIDO = 0.75       # nivel de servicio elegido por el negocio
MIN_FRECUENCIA_NIVEL = 8      # niveles categoricos con menos casos se agrupan

# Minimo de comparables antes de generalizar la busqueda.
#
# Se bajo de 8 a 5 al agregar Marca y Tono a la cascada. Medido sobre los 916
# lanzamientos historicos: exigiendo 8 comparables, el nivel
# 'Canal + Marca + Grupo + Tipo + Tono' solo se alcanzaba en el 9,7% de los
# casos, asi que en la practica Marca y Tono se soltaban casi siempre y el
# resultado era el mismo que antes. Con 5 la cobertura sube de forma util.
#
# El costo es real y hay que tenerlo presente: un P75 calculado sobre 5 puntos
# es ruidoso. Por eso el numero oficial lo da el modelo cuantilico (que usa las
# 916 filas) y la cohorte funciona como respaldo visible y contraste; el
# tamano de la cohorte se reporta siempre en `cohorte_n`.
MIN_COMPARABLES = 5

ABREV_A_NUMERO = {m: i + 1 for i, m in enumerate(
    ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
)}

# Atributos que describen el producto a lanzar
COLUMNAS_CATEGORICAS = ["Canal", "Marca", "Subcategoria", "Grupo", "Tipo", "Tono"]
COLUMNAS_NUMERICAS = ["PVP", "pvp_relativo", "mes_lanzamiento", "lanza_en_S2"]


@dataclass
class ModeloLanzamiento:
    """Modelo entrenado y todo lo necesario para predecir y explicar."""

    modelos_cuantil: dict = field(default_factory=dict)   # {0.25: pipe, 0.50: pipe, ...}
    entrenamiento: pd.DataFrame = None    # los 916 casos historicos, para la cohorte
    mediana_pvp_subcat: pd.Series = None  # para calcular pvp_relativo
    pvp_global: float = 0.0
    metricas: pd.DataFrame = None
    niveles_validos: dict = field(default_factory=dict)
    # Como se reparte la venta entre los 6 primeros meses de vida. Se calcula de
    # los 916 lanzamientos historicos, no se asume plano.
    reparto_mensual: np.ndarray = None

    # --- Horizonte anual (12 meses de rotacion) ---------------------------
    # Modelos y datos propios, entrenados sobre los 686 lanzamientos que tienen
    # 12 meses observables. Van aparte de los de 6 meses a proposito: exigir mas
    # historia reduce el set, y mezclarlos sesgaria el horizonte corto.
    modelos_cuantil_anual: dict = field(default_factory=dict)
    entrenamiento_anual: pd.DataFrame = None
    reparto_mensual_anual: np.ndarray = None
    # Catalogo completo de combinaciones reales de atributos (1.815 registros del
    # maestro, no solo los 916 de entrenamiento). Alimenta los filtros en cascada
    # de la interfaz: el universo de lo que Cosnova efectivamente vende es mas
    # amplio que el de los lanzamientos con historia medible.
    catalogo: pd.DataFrame = None


# ---------------------------------------------------------------------------
# COMBINACIONES VALIDAS (filtros en cascada de la interfaz)
# ---------------------------------------------------------------------------

# Jerarquia de los atributos de producto: cada uno se filtra por los anteriores.
JERARQUIA_ATRIBUTOS = ["Subcategoria", "Grupo", "Tipo", "Tono"]


def opciones_validas(modelo: ModeloLanzamiento, columna: str,
                     seleccion: dict | None = None) -> list[str]:
    """Valores validos de `columna` dadas las selecciones previas del usuario.

    Es lo que permite que al elegir Subcategoria = Labios, el desplegable de
    Grupo solo ofrezca grupos de labios, el de Tipo solo texturas que existen en
    ese grupo, y el de Tono solo tonos que ese producto realmente tiene.

    Sin esto el usuario puede armar combinaciones que no existen en el catalogo
    (por ejemplo un Rubor con tono Transparente), y el modelo se ve forzado a
    predecir sobre algo que nunca ha existido.

    Args:
        modelo: modelo entrenado, con su catalogo.
        columna: atributo cuyas opciones se quieren.
        seleccion: lo ya elegido, p. ej. {"Subcategoria": "Labios", "Grupo": "Labial"}.

    Returns:
        Lista ordenada de valores validos. Si el filtro deja el catalogo vacio,
        devuelve todos los valores de la columna para no bloquear la interfaz.
    """
    catalogo = modelo.catalogo if modelo.catalogo is not None else modelo.entrenamiento
    if catalogo is None or columna not in catalogo.columns:
        return []

    filtrado = catalogo
    for llave, valor in (seleccion or {}).items():
        if llave in filtrado.columns and valor:
            candidato = filtrado[filtrado[llave].astype(str) == str(valor)]
            # Si una seleccion previa no deja nada, se ignora ese filtro en lugar
            # de devolver una lista vacia que dejaria el desplegable inutilizable.
            if not candidato.empty:
                filtrado = candidato

    valores = sorted(filtrado[columna].dropna().astype(str).unique())
    return valores or sorted(catalogo[columna].dropna().astype(str).unique())


def calcular_reparto_mensual(maestro: pd.DataFrame, ent: pd.DataFrame,
                             horizonte: int = MESES_HORIZONTE) -> np.ndarray:
    """Reparto de la venta entre los 6 primeros meses de vida de un lanzamiento.

    Devuelve 6 factores que suman 1. Se calculan de los lanzamientos historicos
    reales, no se asumen.

    **Por que importa.** La curva NO es plana: un lanzamiento construye demanda.
    Medido sobre los 916 casos, el mes 1 concentra el 12,4% de la venta del
    semestre y el mes 6 el 18,6% (ratio 0,67x). Repartir el sugerido dividiendo
    entre 6 sobreestimaria el primer mes en cerca del 35% y subestimaria el
    sexto, justo al contrario de lo que necesita el abastecimiento.

    Se usan medianas normalizadas y no promedios: la distribucion de unidades es
    muy sesgada y unos pocos lanzamientos grandes dominarian el promedio.
    """
    meses = [m for m in cfg.MESES if m in maestro.columns]
    matriz = maestro[meses].to_numpy(dtype=float)

    # Se localiza cada lanzamiento en el maestro por su clave Material + Canal
    indice = {(m, c): i for i, (m, c) in
              enumerate(zip(maestro["Material"], maestro["Canal"]))}

    curvas, absolutas = [], []
    for material, canal, debut in zip(ent["Material"], ent["Canal"], ent["debut_idx"]):
        i = indice.get((material, canal))
        if i is None:
            continue
        tramo = matriz[i, int(debut):int(debut) + horizonte]
        if len(tramo) == horizonte and tramo.sum() > 0:
            curvas.append(tramo / tramo.sum())
            absolutas.append(tramo)

    if not curvas:
        # Sin datos suficientes se reparte plano, que es el supuesto neutro
        return np.full(horizonte, 1 / horizonte)

    factores = np.median(np.array(curvas), axis=0)
    factores = factores / factores.sum()

    return _corregir_primer_mes(np.array(absolutas), factores)


def _corregir_primer_mes(curvas: np.ndarray, factores: np.ndarray) -> np.ndarray:
    """Corrige el sesgo del primer mes medido a medias.

    **El problema.** El mes 1 se define como el primer mes con venta, pero un
    producto que entra al mercado el dia 20 solo vende diez dias de ese mes. Esa
    medicion parcial hunde el promedio y hace parecer que la demanda "arranca
    baja y sube", cuando en realidad el producto ya vende a su ritmo normal.

    **La evidencia de que es un artefacto y no demanda real.** Separando los 686
    lanzamientos historicos segun que tan completo se ve su primer mes:

        Entraron tarde  (mes1/mes2 < 0,5)   n=227   mes 1 = 0,17x el nivel estable
        Intermedios                          n=203   mes 1 = 0,67x
        Entraron temprano (mes1/mes2 >= 0,8) n=233   mes 1 = 1,00x

    Los que si tuvieron un mes completo venden exactamente igual que en los meses
    2 a 6. Ademas la serie salta 2,67x del mes 1 al 2 y despues se queda plana
    (ratios entre 0,79 y 1,31): eso es una discontinuidad puntual, no una rampa.
    Un sell-in inicial habria hecho el mes 1 mas ALTO, no mas bajo, y el efecto
    no depende del mes calendario de lanzamiento, asi que tampoco es estacional.

    **La correccion.** Se estima el nivel de un primer mes completo usando solo
    los lanzamientos que entraron temprano, y se reemplaza el factor del mes 1.
    El resto de meses se reescala para que la curva siga sumando 1: el total
    anual lo fija el modelo y no se infla por este ajuste.
    """
    if curvas.shape[1] < 3 or len(curvas) < MIN_CASOS_PRIMER_MES:
        return factores

    m1, m2 = curvas[:, 0], curvas[:, 1]
    # Nivel de referencia: la mediana de los meses ya estables de cada producto
    n_estables = min(5, curvas.shape[1] - 1)
    base = np.median(curvas[:, 1:1 + n_estables], axis=1)

    utilizable = (m2 > 0) & (base > 0)
    ratio = np.divide(m1, m2, out=np.zeros_like(m1), where=m2 > 0)
    completos = utilizable & (ratio >= UMBRAL_PRIMER_MES_COMPLETO)

    if completos.sum() < MIN_CASOS_PRIMER_MES:
        utils.log("  Pocos lanzamientos con primer mes completo: la curva se deja "
                  "sin corregir", nivel="ALERTA")
        return factores

    factor_completo = float(np.median(m1[completos] / base[completos]))
    nivel_estable = float(factores[1:1 + n_estables].mean())

    corregidos = factores.copy()
    corregidos[0] = nivel_estable * factor_completo
    corregidos = corregidos / corregidos.sum()

    utils.log(f"  Mes 1 corregido: {factores[0]*100:.2f}% -> "
              f"{corregidos[0]*100:.2f}% del total "
              f"(estimado con {int(completos.sum())} lanzamientos de primer mes completo)")

    return corregidos


def sugerido_por_mes(modelo: ModeloLanzamiento, total_6_meses: float,
                     anual: bool = False) -> pd.DataFrame:
    """Reparte el sugerido entre cada mes de rotacion del producto.

    Args:
        total_6_meses: total a repartir (del horizonte que corresponda).
        anual: False reparte en 6 meses; True en 12 meses de rotacion.

    Entrega dos lecturas, porque sirven para cosas distintas:
      - `Promedio plano`:   el total dividido entre los meses. Simple y comparable.
      - `Sugerido del mes`: el total repartido con la curva real de lanzamientos,
                            que arranca mas bajo y sube.
    """
    if anual:
        factores = (modelo.reparto_mensual_anual
                    if modelo.reparto_mensual_anual is not None
                    else np.full(MESES_HORIZONTE_ANUAL, 1 / MESES_HORIZONTE_ANUAL))
        horizonte = MESES_HORIZONTE_ANUAL
        etiqueta = "% del anio"
    else:
        factores = (modelo.reparto_mensual if modelo.reparto_mensual is not None
                    else np.full(MESES_HORIZONTE, 1 / MESES_HORIZONTE))
        horizonte = MESES_HORIZONTE
        etiqueta = "% del semestre"

    total = max(0, int(round(total_6_meses)))
    por_mes = _repartir_entero(total, factores)
    plano = total_6_meses / horizonte

    return pd.DataFrame({
        "Mes de vida": [f"Mes {i + 1}" for i in range(horizonte)],
        etiqueta: (factores * 100).round(1),
        "Sugerido del mes": por_mes,
        "Promedio plano": int(round(plano)),
    })


def _repartir_entero(total: int, factores: np.ndarray) -> np.ndarray:
    """Reparte un entero segun unos factores, conservando la suma exacta.

    Usa el metodo de los restos mayores: se asigna la parte entera de cada
    cuota y las unidades que sobran van a los meses con mayor parte decimal.

    **Por que no redondear cada mes por separado.** Aplicar `ceil` mes a mes
    infla el total: un sugerido de 1 unidad se convertia en 1 por mes, es decir
    6 unidades, contradiciendo el propio total de 6 meses. Con totales pequenos
    el error era enorme en proporcion.
    """
    if total <= 0:
        return np.zeros(len(factores), dtype=int)

    cuotas = total * factores
    base = np.floor(cuotas).astype(int)
    faltan = total - int(base.sum())

    if faltan > 0:
        # Los meses con mayor parte decimal reciben las unidades sobrantes
        orden = np.argsort(-(cuotas - base))
        for i in range(faltan):
            base[orden[i % len(base)]] += 1

    return base


def pvp_tipico(modelo: ModeloLanzamiento, seleccion: dict,
               defecto: float = 20000.0) -> float:
    """PVP mediano de los productos del catalogo que comparten esos atributos.

    Sirve para que el campo de precio de la interfaz arranque en un valor
    plausible para lo que el usuario esta armando, en lugar de un numero fijo.
    Se relajan los filtros de a uno (del mas especifico al mas general) hasta
    encontrar algo con que estimar.
    """
    catalogo = modelo.catalogo if modelo.catalogo is not None else modelo.entrenamiento
    if catalogo is None or "PVP" not in catalogo.columns:
        return defecto

    # Se prueban subconjuntos cada vez mas laxos de la seleccion
    orden = ["Tono", "Tipo", "Grupo", "Subcategoria", "Marca", "Canal"]
    llaves = [k for k in orden if k in seleccion and seleccion[k]]

    for i in range(len(llaves) + 1):
        activos = llaves[i:]
        filtro = pd.Series(True, index=catalogo.index)
        for llave in activos:
            filtro &= catalogo[llave].astype(str) == str(seleccion[llave])
        precios = catalogo.loc[filtro, "PVP"]
        precios = precios[precios > 0]
        if len(precios):
            return float(precios.median())

    return defecto


def combinacion_existe(modelo: ModeloLanzamiento, seleccion: dict) -> int:
    """Cuantos productos del catalogo comparten exactamente esa combinacion.

    Devuelve 0 cuando la combinacion no existe: la interfaz lo usa para avisar
    que se esta consultando algo sin precedente en el portafolio.
    """
    catalogo = modelo.catalogo if modelo.catalogo is not None else modelo.entrenamiento
    if catalogo is None:
        return 0

    filtro = pd.Series(True, index=catalogo.index)
    for llave, valor in seleccion.items():
        if llave in catalogo.columns and valor:
            filtro &= catalogo[llave].astype(str) == str(valor)
    return int(filtro.sum())


# ---------------------------------------------------------------------------
# 1. CONSTRUCCION DEL SET DE ENTRENAMIENTO
# ---------------------------------------------------------------------------


def identificar_debut(df: pd.DataFrame) -> pd.DataFrame:
    """Marca el primer mes con venta de cada registro producto x canal."""
    meses = [m for m in cfg.MESES if m in df.columns]
    matriz = df[meses].to_numpy()
    tiene_venta = matriz > 0

    df = df.copy()
    # argmax sobre booleanos devuelve el primer True; -1 si nunca vendio
    df["debut_idx"] = np.where(tiene_venta.any(axis=1), tiene_venta.argmax(axis=1), -1)
    df["debut_mes"] = [meses[i] if i >= 0 else None for i in df["debut_idx"]]
    return df


def construir_entrenamiento(maestro: pd.DataFrame | None = None,
                            horizonte: int = MESES_HORIZONTE) -> pd.DataFrame:
    """Arma el set de lanzamientos historicos con su target medible.

    Args:
        maestro: dataset unificado; se carga solo si no se entrega.
        horizonte: meses de vida a medir. 6 para la primera compra semestral,
            12 para el horizonte anual. Exigir mas meses observables reduce el
            set: 916 lanzamientos con 6 meses, 686 con 12.
    """
    utils.log(f"Construyendo set de entrenamiento a {horizonte} meses ...")

    if maestro is None:
        maestro = ingesta.cargar_maestro()

    meses = [m for m in cfg.MESES if m in maestro.columns]
    matriz = maestro[meses].to_numpy()
    df = identificar_debut(maestro).reset_index(drop=True)

    # Solo debuts limpios: posteriores al primer mes de la ventana (no censurados)
    # y con el horizonte completo observable.
    limite = len(meses) - horizonte
    valido = (df["debut_idx"] >= 1) & (df["debut_idx"] <= limite)

    n_nunca = int((df["debut_idx"] < 0).sum())
    n_censura = int((df["debut_idx"] == 0).sum())
    n_reciente = int((df["debut_idx"] > limite).sum())

    utils.log(f"  Nunca vendieron (a predecir):      {n_nunca:>5}")
    utils.log(f"  Censurados: ya vendian en {meses[0]}:  {n_censura:>5}  (excluidos)")
    utils.log(f"  Debut hace menos de {horizonte} meses:    {n_reciente:>5}  (excluidos)")
    utils.log(f"  Set de entrenamiento:             {int(valido.sum()):>5}")

    ent = df.loc[valido].copy()

    # Target: unidades en los primeros `horizonte` meses desde el debut.
    # La columna mantiene el nombre 'und_primeros_6m' en ambos horizontes porque
    # es la que consumen la cohorte de analogos y los reportes; asi el resto del
    # modulo no necesita saber con que horizonte se construyo el set.
    ent["und_primeros_6m"] = [
        matriz[i, int(d):int(d) + horizonte].sum()
        for i, d in zip(ent.index, ent["debut_idx"])
    ]

    # Mes calendario del lanzamiento: captura que lanzar en julio no es igual que
    # lanzar en enero (el segundo semestre vende mas).
    ent["mes_lanzamiento"] = [ABREV_A_NUMERO[m.split("-")[0]] for m in ent["debut_mes"]]
    ent["lanza_en_S2"] = (ent["mes_lanzamiento"] >= 7).astype(int)

    utils.log(f"  Target: mediana {ent['und_primeros_6m'].median():.0f} und | "
              f"P75 {ent['und_primeros_6m'].quantile(0.75):.0f} und | "
              f"max {ent['und_primeros_6m'].max():.0f} und")

    return ent.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURES
# ---------------------------------------------------------------------------


def calcular_pvp_relativo(df: pd.DataFrame, mediana_subcat: pd.Series,
                          pvp_global: float) -> pd.Series:
    """PVP del producto dividido por la mediana de PVP de su subcategoria.

    Un labial de 23.000 no es "caro" en absoluto: es caro o barato *frente a los
    otros labiales*. Esta variable resulto mas informativa que el PVP crudo
    (0,074 vs 0,033 de importancia en el modelo).
    """
    referencia = df["Subcategoria"].map(mediana_subcat).fillna(pvp_global)
    return (df["PVP"] / referencia.replace(0, np.nan)).fillna(1.0)


def preparar_features(df: pd.DataFrame, modelo: ModeloLanzamiento) -> pd.DataFrame:
    """Arma la matriz de entrada, identica en entrenamiento y prediccion."""
    X = pd.DataFrame(index=df.index)
    for columna in COLUMNAS_CATEGORICAS:
        X[columna] = df[columna].astype(str)
    X["PVP"] = pd.to_numeric(df["PVP"], errors="coerce").fillna(modelo.pvp_global)
    X["pvp_relativo"] = calcular_pvp_relativo(df, modelo.mediana_pvp_subcat,
                                              modelo.pvp_global)
    X["mes_lanzamiento"] = pd.to_numeric(df["mes_lanzamiento"], errors="coerce").fillna(7)
    X["lanza_en_S2"] = (X["mes_lanzamiento"] >= 7).astype(int)
    return X


def construir_pipeline(alpha: float) -> Pipeline:
    """Pipeline de One-Hot + Gradient Boosting cuantilico.

    Se usa un Pipeline con ColumnTransformer (y no get_dummies manual) para que
    entrenamiento y prediccion compartan exactamente las mismas columnas y el
    mismo orden. Con get_dummies, un producto nuevo con un Tono no visto generaria
    una matriz de forma distinta y la prediccion fallaria o, peor, se desalinearia
    en silencio.

    `handle_unknown="ignore"` permite predecir un producto con un atributo nunca
    visto: esa columna queda en cero y el modelo se apoya en el resto.
    `min_frequency` agrupa los niveles raros, evitando dimensiones casi vacias.
    """
    codificador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=MIN_FRECUENCIA_NIVEL,
                                  sparse_output=False),
             COLUMNAS_CATEGORICAS),
        ],
        remainder="passthrough",
    )
    regresor = GradientBoostingRegressor(
        loss="quantile", alpha=alpha,
        n_estimators=300, max_depth=3, learning_rate=0.05,
        random_state=cfg.SEMILLA,
    )
    return Pipeline([("codificar", codificador), ("modelo", regresor)])


# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO Y VALIDACION
# ---------------------------------------------------------------------------


def validar_modelo(ent: pd.DataFrame, modelo: ModeloLanzamiento) -> pd.DataFrame:
    """Compara el modelo contra baselines con GroupKFold por Material.

    Se reportan tres metricas porque ninguna sola basta:
      - MAE: error en unidades, legible para el negocio.
      - MAE_log: error en escala logaritmica, robusto a la cola larga. Es la que
        refleja el error *relativo*, que es lo que importa cuando el target va de
        1 a 7.767 unidades.
      - dentro_factor2: % de casos donde la prediccion queda entre la mitad y el
        doble del real. Es la lectura mas honesta para planeacion.
    """
    utils.log("Validando con GroupKFold por Material (5 folds) ...")

    X = preparar_features(ent, modelo)
    y = ent["und_primeros_6m"].to_numpy(dtype=float)
    grupos = ent["Material"].to_numpy()
    canal = ent["Canal"].to_numpy()
    cv = GroupKFold(n_splits=5)

    def metricas(reales, predichos):
        p = np.clip(predichos, 0, None)
        return {
            "MAE": round(float(np.mean(np.abs(reales - p))), 1),
            "MAE_log": round(float(np.mean(np.abs(np.log1p(reales) - np.log1p(p)))), 4),
            "dentro_factor2_%": round(float(np.mean((p >= reales / 2) & (p <= reales * 2)) * 100), 1),
        }

    filas = []

    # Baseline: mediana por canal. Es el metodo que un analista usaria sin modelo,
    # asi que es la vara real que hay que superar.
    reales, preds = [], []
    for tr, te in cv.split(X, y, groups=grupos):
        mediana = pd.Series(y[tr]).groupby(canal[tr]).median()
        global_ = np.median(y[tr])
        preds.append(np.array([mediana.get(c, global_) for c in canal[te]]))
        reales.append(y[te])
    filas.append({"metodo": "Baseline: mediana por canal",
                  **metricas(np.concatenate(reales), np.concatenate(preds))})

    # RandomForest sobre log1p: mide cuanta senal aportan los atributos
    reales, preds = [], []
    for tr, te in cv.split(X, y, groups=grupos):
        pipe = Pipeline([
            ("codificar", ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                       min_frequency=MIN_FRECUENCIA_NIVEL,
                                       sparse_output=False), COLUMNAS_CATEGORICAS)],
                remainder="passthrough")),
            ("modelo", RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                             random_state=cfg.SEMILLA, n_jobs=-1)),
        ])
        pipe.fit(X.iloc[tr], np.log1p(y[tr]))
        preds.append(np.expm1(pipe.predict(X.iloc[te])))
        reales.append(y[te])
    filas.append({"metodo": "RandomForest (atributos)",
                  **metricas(np.concatenate(reales), np.concatenate(preds))})

    tabla = pd.DataFrame(filas)
    mejora = (1 - tabla.loc[1, "MAE_log"] / tabla.loc[0, "MAE_log"]) * 100
    print(tabla.to_string(index=False))
    utils.log(f"Mejora del modelo sobre el baseline: {mejora:+.1f}% en MAE_log")

    if mejora < 10:
        utils.log("La ganancia sobre el baseline es marginal: preferir la cohorte "
                  "de analogos, que es mas interpretable.", nivel="ALERTA")

    return tabla


def verificar_calibracion(ent: pd.DataFrame, modelo: ModeloLanzamiento) -> pd.DataFrame:
    """Comprueba que el P75 cubra de verdad cerca del 75% de los casos.

    Un cuantil mal calibrado es peor que no tenerlo: si el "P75" solo cubre el 55%
    real, el comprador cree tener colchon y no lo tiene.
    """
    utils.log("Verificando calibracion de los cuantiles ...")

    X = preparar_features(ent, modelo)
    y = ent["und_primeros_6m"].to_numpy(dtype=float)
    grupos = ent["Material"].to_numpy()
    cv = GroupKFold(n_splits=5)

    filas = []
    for alpha in CUANTILES:
        cubierto = []
        for tr, te in cv.split(X, y, groups=grupos):
            pipe = construir_pipeline(alpha)
            pipe.fit(X.iloc[tr], np.log1p(y[tr]))
            prediccion = np.expm1(pipe.predict(X.iloc[te]))
            cubierto.append(y[te] <= prediccion)
        real = float(np.concatenate(cubierto).mean() * 100)
        filas.append({"cuantil": alpha, "cobertura_objetivo_%": alpha * 100,
                      "cobertura_real_%": round(real, 1),
                      "desvio_pp": round(real - alpha * 100, 1)})
        utils.log(f"  P{int(alpha*100)}: cubre {real:.1f}% de los casos "
                  f"(objetivo {alpha*100:.0f}%)")

    return pd.DataFrame(filas)


def entrenar(maestro: pd.DataFrame | None = None,
             validar: bool = True) -> ModeloLanzamiento:
    """Entrena los tres modelos cuantilicos y devuelve el objeto listo para usar."""
    utils.titulo("MODELO DE SUGERIDO PARA LANZAMIENTOS")

    if maestro is None:
        maestro = ingesta.cargar_maestro()

    ent = construir_entrenamiento(maestro)

    # Catalogo de combinaciones reales de atributos, tomado del maestro completo
    # y no solo del set de entrenamiento: alimenta los filtros en cascada de la
    # interfaz. Se conserva Canal para poder filtrar tambien por canal.
    columnas_catalogo = [c for c in ["Canal", "Marca", "Subcategoria", "Grupo",
                                     "Tipo", "Tono", "PVP"] if c in maestro.columns]
    catalogo = maestro[columnas_catalogo].drop_duplicates().reset_index(drop=True)

    modelo = ModeloLanzamiento(
        entrenamiento=ent,
        mediana_pvp_subcat=ent.groupby("Subcategoria")["PVP"].median(),
        pvp_global=float(ent["PVP"].median()),
        # Los niveles validos salen del CATALOGO y no del entrenamiento: el
        # usuario debe poder elegir cualquier atributo que Cosnova venda, aunque
        # todavia no haya un lanzamiento historico con esa combinacion.
        niveles_validos={c: sorted(catalogo[c].dropna().astype(str).unique())
                         for c in COLUMNAS_CATEGORICAS if c in catalogo.columns},
        catalogo=catalogo,
        reparto_mensual=calcular_reparto_mensual(maestro, ent),
    )
    utils.log(f"  Catalogo de combinaciones: {len(catalogo)} filas "
              f"(para los filtros en cascada)")
    reparto = " | ".join(f"M{i+1}:{f*100:.1f}%"
                         for i, f in enumerate(modelo.reparto_mensual))
    utils.log(f"  Reparto mensual del debut: {reparto}")

    if validar:
        modelo.metricas = validar_modelo(ent, modelo)
        modelo.calibracion = verificar_calibracion(ent, modelo)

    utils.log("Entrenando modelos cuantilicos a 6 meses ...")
    X = preparar_features(ent, modelo)
    y_log = np.log1p(ent["und_primeros_6m"].to_numpy(dtype=float))

    for alpha in CUANTILES:
        pipe = construir_pipeline(alpha)
        pipe.fit(X, y_log)
        modelo.modelos_cuantil[alpha] = pipe
        utils.log(f"  P{int(alpha*100)} entrenado")

    # --- Horizonte anual: modelos propios sobre su propio set --------------
    utils.log(f"Entrenando modelos cuantilicos a {MESES_HORIZONTE_ANUAL} meses ...")
    ent_anual = construir_entrenamiento(maestro, horizonte=MESES_HORIZONTE_ANUAL)
    modelo.entrenamiento_anual = ent_anual
    modelo.reparto_mensual_anual = calcular_reparto_mensual(
        maestro, ent_anual, horizonte=MESES_HORIZONTE_ANUAL)

    X_anual = preparar_features(ent_anual, modelo)
    y_anual = np.log1p(ent_anual["und_primeros_6m"].to_numpy(dtype=float))
    for alpha in CUANTILES:
        pipe = construir_pipeline(alpha)
        pipe.fit(X_anual, y_anual)
        modelo.modelos_cuantil_anual[alpha] = pipe
        utils.log(f"  P{int(alpha*100)} anual entrenado")

    reparto = " | ".join(f"M{i+1}:{f*100:.1f}%"
                         for i, f in enumerate(modelo.reparto_mensual_anual))
    utils.log(f"  Reparto a 12 meses: {reparto}")
    s1 = modelo.reparto_mensual_anual[:6].sum()
    utils.log(f"  Meses 1-6: {s1*100:.1f}% del anio | "
              f"Meses 7-12: {(1-s1)*100:.1f}%")

    return modelo


# ---------------------------------------------------------------------------
# 4. COHORTE DE ANALOGOS
# ---------------------------------------------------------------------------

# Cascada de busqueda: de lo mas especifico a lo mas general. Se baja un nivel
# solo cuando no hay suficientes comparables. El nivel alcanzado es la medida de
# confianza del resultado.
#
# ORDEN EN QUE SE SUELTAN LOS ATRIBUTOS (y por que ese orden)
#
# El criterio no es arbitrario: se midio cuanto cambia la venta de los primeros
# 6 meses al variar cada atributo, sobre los 916 lanzamientos historicos.
#
#   Canal  -> se conserva SIEMPRE. Explica el 47% de la prediccion; la mediana de
#             debut es 709 und en Farmatodo contra 69 en Prosalon (10x).
#   Tono   -> se suelta TARDE. Controlando por grupo, la mediana varia 2-3x
#             (Brillo 3,0x | Pestanina 2,8x | Rubor 2,3x | Labial 2,1x).
#             Ojo: la direccion depende del grupo. Dentro de Brillo el cafe
#             vende MAS que el rosado (606 vs 267); globalmente es al reves
#             (176 vs 239) porque el rosado abunda en Rubor.
#   Grupo  -> se suelta tarde: es la definicion funcional del producto.
#   Marca  -> se suelta antes que Tono y Grupo, pero se incluye: comparar un
#             lanzamiento Essence contra productos Catrice desorienta al
#             comprador, aunque el efecto medido de la marca sea menor.
#   Tipo   -> se suelta primero entre los de producto: es la textura y resulto
#             el menos discriminante de los cuatro.
#   PVP    -> solo afina los dos niveles mas especificos.
NIVELES_COHORTE = [
    # --- Maxima especificidad: marca, grupo, tipo, tono y precio ------------
    ("Canal + Marca + Grupo + Tipo + Tono + PVP +-20%",
     ["Canal", "Marca", "Grupo", "Tipo", "Tono"], 0.20),
    ("Canal + Marca + Grupo + Tipo + Tono",
     ["Canal", "Marca", "Grupo", "Tipo", "Tono"], None),
    # --- Se suelta Tipo, se conservan Marca y Tono -------------------------
    ("Canal + Marca + Grupo + Tono",
     ["Canal", "Marca", "Grupo", "Tono"], None),
    # --- Se suelta Marca, se conserva Tono --------------------------------
    ("Canal + Grupo + Tipo + Tono",
     ["Canal", "Grupo", "Tipo", "Tono"], None),
    ("Canal + Grupo + Tono",
     ["Canal", "Grupo", "Tono"], None),
    # --- Ya sin Tono: se vuelve a intentar con Marca ----------------------
    ("Canal + Marca + Grupo + Tipo",
     ["Canal", "Marca", "Grupo", "Tipo"], None),
    ("Canal + Marca + Grupo",
     ["Canal", "Marca", "Grupo"], None),
    ("Canal + Grupo + Tipo",
     ["Canal", "Grupo", "Tipo"], None),
    ("Canal + Grupo",
     ["Canal", "Grupo"], None),
    # --- Ultimo recurso: subcategoria, priorizando el tono ----------------
    ("Canal + Subcategoria + Tono",
     ["Canal", "Subcategoria", "Tono"], None),
    ("Canal + Subcategoria + Tipo",
     ["Canal", "Subcategoria", "Tipo"], None),
    ("Canal + Subcategoria",
     ["Canal", "Subcategoria"], None),
    ("Canal",
     ["Canal"], None),
]


def buscar_cohorte(modelo: ModeloLanzamiento, atributos: dict,
                   minimo: int = MIN_COMPARABLES) -> tuple[pd.DataFrame, str]:
    """Busca productos historicos comparables, generalizando si hace falta.

    Devuelve (comparables, nivel_usado). El nivel indica cuan especifica pudo ser
    la busqueda: "Canal + Marca + Grupo + Tipo + Tono" es evidencia fuerte,
    "Canal" a secas significa que no habia nada parecido y el numero es apenas
    una referencia.

    Los comparables salen ordenados con los de la MISMA marca primero. Cuando la
    cascada tuvo que soltar la marca, el comprador igual ve arriba lo que si
    coincide con su producto, en lugar de una lista encabezada por la otra marca.
    """
    base = modelo.entrenamiento

    for descripcion, llaves, tolerancia_pvp in NIVELES_COHORTE:
        filtro = pd.Series(True, index=base.index)

        for llave in llaves:
            valor = atributos.get(llave)
            if valor is None:
                filtro &= False
                break
            filtro &= base[llave].astype(str).str.lower() == str(valor).lower()

        if tolerancia_pvp is not None and atributos.get("PVP"):
            pvp = float(atributos["PVP"])
            filtro &= base["PVP"].between(pvp * (1 - tolerancia_pvp),
                                          pvp * (1 + tolerancia_pvp))

        comparables = base.loc[filtro]
        if len(comparables) >= minimo:
            return _ordenar_comparables(comparables, atributos), descripcion

    # Ni con el canal solo hay suficientes: se devuelve todo el historico
    return _ordenar_comparables(base, atributos), \
        "Todo el historico (sin comparables especificos)"


def _ordenar_comparables(comparables: pd.DataFrame, atributos: dict) -> pd.DataFrame:
    """Ordena los comparables por parecido al producto consultado.

    Prioridad: misma marca, luego mismo tono, luego mayor venta. Es un orden de
    presentacion, no afecta el calculo (los cuantiles se computan sobre toda la
    cohorte); sirve para que lo primero que vea el comprador sea lo mas parecido
    a lo que va a lanzar.
    """
    if comparables.empty:
        return comparables

    salida = comparables.copy()
    marca = str(atributos.get("Marca", "")).lower()
    tono = str(atributos.get("Tono", "")).lower()

    salida["_misma_marca"] = (salida["Marca"].astype(str).str.lower() == marca).astype(int)
    salida["_mismo_tono"] = (salida["Tono"].astype(str).str.lower() == tono).astype(int)

    return (salida.sort_values(["_misma_marca", "_mismo_tono", "und_primeros_6m"],
                               ascending=[False, False, False])
                  .drop(columns=["_misma_marca", "_mismo_tono"]))


def diagnostico_cohorte(comparables: pd.DataFrame, nivel: str,
                        atributos: dict) -> dict:
    """Informa que atributos quedaron efectivamente respetados en la cohorte.

    Sin esto el usuario no puede saber si los comparables que ve corresponden a
    su marca y su tono, que es exactamente la confusion que genera mostrar
    productos Catrice cuando se consulto un Essence.
    """
    if comparables.empty:
        return {"marca_respetada": False, "tono_respetado": False,
                "pct_misma_marca": 0.0, "pct_mismo_tono": 0.0}

    marca = str(atributos.get("Marca", "")).lower()
    tono = str(atributos.get("Tono", "")).lower()
    misma_marca = comparables["Marca"].astype(str).str.lower() == marca
    mismo_tono = comparables["Tono"].astype(str).str.lower() == tono

    return {
        # 'Respetado' = el atributo formo parte del criterio de busqueda, asi que
        # TODOS los comparables lo comparten.
        "marca_respetada": "Marca" in nivel,
        "tono_respetado": "Tono" in nivel,
        "pct_misma_marca": round(100 * float(misma_marca.mean()), 1),
        "pct_mismo_tono": round(100 * float(mismo_tono.mean()), 1),
    }


# ---------------------------------------------------------------------------
# 5. PREDICCION
# ---------------------------------------------------------------------------


def _normalizar_entrada(modelo: ModeloLanzamiento, columna: str, valor):
    """Resuelve el valor que escribio el usuario contra los niveles reales.

    Tolera diferencias de mayusculas y tildes ('liquido' -> 'Liquido'). Si no
    encuentra el nivel, avisa y devuelve el valor tal cual: el modelo lo tratara
    como categoria no vista, que es el comportamiento correcto y no un error.
    """
    if valor is None:
        return None
    validos = modelo.niveles_validos.get(columna, [])
    objetivo = utils.normalizar_texto(valor)

    for nivel in validos:
        if utils.normalizar_texto(nivel) == objetivo:
            return nivel

    if validos:
        utils.log(f"'{valor}' no existe en '{columna}'. Valores validos: "
                  f"{', '.join(map(str, validos[:12]))}"
                  f"{' ...' if len(validos) > 12 else ''}", nivel="ALERTA")
    return valor


def sugerir_lanzamiento(
    modelo: ModeloLanzamiento,
    canal: str,
    subcategoria: str,
    grupo: str,
    tipo: str,
    pvp: float,
    tono: str = "Sin dato",
    marca: str = "Essence",
    mes_lanzamiento: int = 7,
    mostrar: bool = True,
) -> dict:
    """Calcula el sugerido de unidades para un producto a lanzar.

    Args:
        canal: Exito, Farmatodo o Prosalon.
        subcategoria: Labios, Rostro, Ojos, Unas, Cejas.
        grupo: Labial, Brillo, Rubor, Pestanina, ...
        tipo: Barra, Liquido, Cremoso, Polvo, Lapiz, ...
        pvp: precio de venta al publico en pesos.
        tono: Rojo, Nude, Negro, ...
        marca: Essence o Catrice.
        mes_lanzamiento: mes calendario del lanzamiento (1-12). Julio por defecto,
            que es el arranque del segundo semestre.

    Returns:
        dict con P25/P50/P75, el sugerido, la cohorte de analogos y la senal de
        revision si el modelo y la cohorte no coinciden.
    """
    atributos = {
        "Canal": _normalizar_entrada(modelo, "Canal", canal),
        "Marca": _normalizar_entrada(modelo, "Marca", marca),
        "Subcategoria": _normalizar_entrada(modelo, "Subcategoria", subcategoria),
        "Grupo": _normalizar_entrada(modelo, "Grupo", grupo),
        "Tipo": _normalizar_entrada(modelo, "Tipo", tipo),
        "Tono": _normalizar_entrada(modelo, "Tono", tono),
        "PVP": float(pvp),
        "mes_lanzamiento": int(mes_lanzamiento),
    }

    fila = pd.DataFrame([atributos])
    X = preparar_features(fila, modelo)

    # --- Prediccion del modelo cuantilico ---------------------------------
    predicciones = {}
    for alpha, pipe in modelo.modelos_cuantil.items():
        valor = float(np.expm1(pipe.predict(X)[0]))
        predicciones[alpha] = max(0.0, valor)

    # Los cuantiles se entrenan por separado y pueden cruzarse en zonas con pocos
    # datos (un P50 mayor que el P75). Se ordenan para que el resultado sea
    # siempre coherente al leerlo.
    ordenados = sorted(predicciones.values())
    p25, p50, p75 = ordenados[0], ordenados[1], ordenados[2]

    # --- Cohorte de analogos ----------------------------------------------
    comparables, nivel = buscar_cohorte(modelo, atributos)
    coh_p25, coh_p50, coh_p75 = comparables["und_primeros_6m"].quantile([.25, .5, .75])

    sugerido = int(np.ceil(p75))
    sugerido_cohorte = int(np.ceil(coh_p75))

    # --- Senal de divergencia ---------------------------------------------
    # Si modelo y evidencia directa difieren por mas del doble, el numero no es
    # confiable: puede ser una combinacion de atributos sin respaldo historico.
    mayor = max(sugerido, sugerido_cohorte, 1)
    menor = max(min(sugerido, sugerido_cohorte), 1)
    divergencia = mayor / menor
    revisar = divergencia > 2.0

    diagnostico = diagnostico_cohorte(comparables, nivel, atributos)

    resultado = {
        "atributos": atributos,
        "P25": round(p25, 1), "P50": round(p50, 1), "P75": round(p75, 1),
        "unidades_sugeridas": sugerido,
        "cohorte_nivel": nivel,
        "cohorte_n": len(comparables),
        **diagnostico,
        "cohorte_P25": round(float(coh_p25), 1),
        "cohorte_P50": round(float(coh_p50), 1),
        "cohorte_P75": round(float(coh_p75), 1),
        "cohorte_sugerido": sugerido_cohorte,
        "divergencia": round(divergencia, 2),
        "requiere_revision": bool(revisar),
        "ingreso_proyectado": round(sugerido * float(pvp), 0),
        "comparables": comparables,
    }

    if mostrar:
        _imprimir_sugerido(resultado)

    return resultado


def _imprimir_sugerido(r: dict) -> None:
    """Muestra el resultado en formato legible para el analista."""
    a = r["atributos"]
    utils.titulo("SUGERIDO DE LANZAMIENTO")
    print(f"  Producto:  {a['Marca']} | {a['Subcategoria']} | {a['Grupo']} | "
          f"{a['Tipo']} | Tono {a['Tono']}")
    print(f"  Canal:     {a['Canal']}")
    print(f"  PVP:       $ {utils.formatear_miles(a['PVP'])}")
    print(f"  Lanza en:  mes {a['mes_lanzamiento']}"
          f"{'  (segundo semestre)' if a['mes_lanzamiento'] >= 7 else '  (primer semestre)'}")

    print(f"\n  ESTIMACION PARA LOS PRIMEROS {MESES_HORIZONTE} MESES")
    print(f"    Escenario bajo   (P25): {utils.formatear_miles(r['P25']):>8} unidades")
    print(f"    Escenario medio  (P50): {utils.formatear_miles(r['P50']):>8} unidades")
    print(f"    Escenario alto   (P75): {utils.formatear_miles(r['P75']):>8} unidades")
    print(f"\n  >>> SUGERIDO DE COMPRA: {utils.formatear_miles(r['unidades_sugeridas'])} "
          f"unidades  (P{int(CUANTIL_SUGERIDO*100)}, nivel de servicio del negocio)")
    print(f"      Ingreso proyectado: $ {utils.formatear_miles(r['ingreso_proyectado'])}")

    print(f"\n  RESPALDO: {r['cohorte_n']} productos comparables")
    print(f"    Criterio de busqueda: {r['cohorte_nivel']}")
    print(f"    Su venta real: P25={utils.formatear_miles(r['cohorte_P25'])} | "
          f"P50={utils.formatear_miles(r['cohorte_P50'])} | "
          f"P75={utils.formatear_miles(r['cohorte_P75'])}")
    detalle_marca = ("coincide en todos" if r["marca_respetada"]
                     else f"solo en {r['pct_misma_marca']}% de los comparables")
    detalle_tono = ("coincide en todos" if r["tono_respetado"]
                    else f"solo en {r['pct_mismo_tono']}% de los comparables")
    print(f"    Marca {a['Marca']}: {detalle_marca}")
    print(f"    Tono {a['Tono']}: {detalle_tono}")

    if not r["marca_respetada"] or not r["tono_respetado"]:
        sueltos = [x for x, ok in [("marca", r["marca_respetada"]),
                                   ("tono", r["tono_respetado"])] if not ok]
        print(f"    [NOTA] No habia suficientes comparables conservando "
              f"{' y '.join(sueltos)}: la busqueda tuvo que ampliarse.")

    if r["requiere_revision"]:
        print(f"\n  [ALERTA] El modelo ({utils.formatear_miles(r['unidades_sugeridas'])}) y "
              f"los comparables ({utils.formatear_miles(r['cohorte_sugerido'])}) difieren "
              f"{r['divergencia']}x.")
        print("           Combinacion de atributos con poco respaldo historico: "
              "validar con el area comercial antes de pedir.")
    else:
        print(f"\n  Modelo y comparables coinciden (diferencia {r['divergencia']}x): "
              "estimacion respaldada por la evidencia.")

    comparables = r["comparables"]
    if len(comparables):
        print("\n  Productos comparables mas parecidos (venta real en sus primeros "
              f"{MESES_HORIZONTE} meses):")
        muestra = comparables.nlargest(min(8, len(comparables)), "und_primeros_6m")
        for _, fila in muestra.iterrows():
            print(f"    {str(fila['Descripcion'])[:42]:<42} "
                  f"PVP {fila['PVP']:>7.0f}  "
                  f"{utils.formatear_miles(fila['und_primeros_6m']):>7} und  "
                  f"(debut {fila['debut_mes']})")


# ---------------------------------------------------------------------------
# 6. SCORING MASIVO DE LOS LANZAMIENTOS PENDIENTES
# ---------------------------------------------------------------------------


def puntuar_pendientes(modelo: ModeloLanzamiento,
                       maestro: pd.DataFrame | None = None,
                       mes_lanzamiento: int = 7) -> pd.DataFrame:
    """Calcula el sugerido para todos los productos sin historial del archivo.

    Son los 385 registros producto x canal que no vendieron una sola unidad en 30
    meses; en su mayoria son novedades 2026 que todavia no han entrado a vender.
    El pipeline principal los deja en "definir comercialmente"; aqui si reciben
    una cifra estimada desde sus atributos.
    """
    utils.titulo("SUGERIDO PARA LOS LANZAMIENTOS PENDIENTES")

    if maestro is None:
        maestro = ingesta.cargar_maestro()

    meses = [m for m in cfg.MESES if m in maestro.columns]
    pendientes = maestro.loc[maestro[meses].sum(axis=1) == 0].copy()
    utils.log(f"Registros sin historial a puntuar: {len(pendientes):,} "
              f"({pendientes['Material'].nunique()} materiales unicos)")

    pendientes["mes_lanzamiento"] = mes_lanzamiento
    X = preparar_features(pendientes, modelo)

    for alpha, pipe in modelo.modelos_cuantil.items():
        pendientes[f"P{int(alpha*100)}"] = np.clip(np.expm1(pipe.predict(X)), 0, None).round(1)

    # Se ordenan los cuantiles fila a fila: entrenados por separado pueden cruzarse
    columnas_q = ["P25", "P50", "P75"]
    ordenado = np.sort(pendientes[columnas_q].to_numpy(), axis=1)
    pendientes[columnas_q] = ordenado.round(1)

    pendientes["unidades_sugeridas_lanzamiento"] = np.ceil(pendientes["P75"]).astype("int64")
    pendientes["ingreso_proyectado_lanzamiento"] = (
        pendientes["unidades_sugeridas_lanzamiento"] * pendientes["PVP"]
    ).round(0)

    # Mismo horizonte anual y desagregado mensual que en la carga de un Excel
    pendientes = agregar_horizonte_anual(
        pendientes, X, modelo, col_semestral="unidades_sugeridas_lanzamiento")

    # Cohorte por fila: da el nivel de respaldo y la senal de revision
    niveles, n_comp, coh_p75, revisar = [], [], [], []
    for _, fila in pendientes.iterrows():
        atributos = {c: fila[c] for c in COLUMNAS_CATEGORICAS}
        atributos["PVP"] = fila["PVP"]
        comparables, nivel = buscar_cohorte(modelo, atributos)
        p75_coh = float(comparables["und_primeros_6m"].quantile(0.75))
        niveles.append(nivel)
        n_comp.append(len(comparables))
        coh_p75.append(round(p75_coh, 1))
        mayor = max(fila["unidades_sugeridas_lanzamiento"], p75_coh, 1)
        menor = max(min(fila["unidades_sugeridas_lanzamiento"], p75_coh), 1)
        revisar.append(bool(mayor / menor > 2.0))

    pendientes["cohorte_nivel"] = niveles
    pendientes["cohorte_n"] = n_comp
    pendientes["cohorte_P75"] = coh_p75
    pendientes["requiere_revision"] = revisar

    utils.log(f"Unidades sugeridas totales: "
              f"{utils.formatear_miles(pendientes['unidades_sugeridas_lanzamiento'].sum())}")
    utils.log(f"Ingreso proyectado: $ "
              f"{utils.formatear_miles(pendientes['ingreso_proyectado_lanzamiento'].sum())}")
    utils.log(f"Registros que requieren revision manual: "
              f"{int(pendientes['requiere_revision'].sum())} "
              f"({utils.porcentaje(int(pendientes['requiere_revision'].sum()), len(pendientes))}%)")

    print("\nSugerido por canal:")
    print(pendientes.groupby("Canal").agg(
        registros=("Material", "count"),
        sugerido_total=("unidades_sugeridas_lanzamiento", "sum"),
        sugerido_mediano=("unidades_sugeridas_lanzamiento", "median"),
    ).to_string())

    columnas = [
        "Material", "EAN/UPC-Code", "Descripcion", "Marca", "Canal", "Status",
        "Tipo Portafolio", "Subcategoria", "Grupo", "Tono", "Tipo", "PVP",
        # Horizonte semestral
        "P25", "P50", "P75", "unidades_sugeridas_lanzamiento",
        "ingreso_proyectado_lanzamiento",
        # Horizonte anual (12 meses de rotacion)
        "P25_anual", "P50_anual", "P75_anual",
        "unidades_sugeridas_anual", "ingreso_proyectado_anual",
        # Desagregado mes a mes
        *COLUMNAS_MES,
        # Respaldo
        "cohorte_nivel", "cohorte_n", "cohorte_P75", "requiere_revision",
    ]
    existentes = [c for c in columnas if c in pendientes.columns]
    return (pendientes[existentes]
            .sort_values(["Canal", "unidades_sugeridas_lanzamiento"], ascending=[True, False])
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 6-bis. CARGA MASIVA DESDE EXCEL
# ---------------------------------------------------------------------------

# Alias de encabezado aceptados. La busqueda es por coincidencia EXACTA sobre el
# nombre normalizado (minusculas, sin tildes, espacios colapsados).
#
# La coincidencia exacta es deliberada y no un atajo: con busqueda parcial,
# 'Tipo Portafolio' capturaria el alias 'tipo' y el modelo recibiria
# "New's Junio 2026" como si fuera la textura del producto. Ese error seria
# silencioso y arruinaria todas las predicciones sin dar ninguna senal.
ALIAS_COLUMNAS = {
    "Canal": ["canal", "canal de distribucion", "cliente", "punto de venta"],
    "Marca": ["marca", "brand"],
    "Subcategoria": ["subcategoria", "sub categoria", "sub-categoria", "linea"],
    "Grupo": ["grupo", "grupo de producto", "categoria de producto"],
    "Tipo": ["tipo", "textura", "formato"],
    "Tono": ["tono", "color", "shade"],
    "PVP": ["pvp", "precio", "precio venta publico", "precio de venta",
            "pvp sugerido", "precio publico"],
    "mes_lanzamiento": ["mes lanzamiento", "mes_lanzamiento", "mes de lanzamiento",
                        "mes", "mes lanza"],
    "Descripcion": ["descripcion", "producto", "nombre", "nombre producto",
                    "descripcion producto"],
    "Material": ["material", "sku", "codigo", "codigo material", "cod material"],
    "EAN/UPC-Code": ["ean/upc-code", "ean", "upc", "ean upc code", "codigo de barras"],
}

# Sin estas columnas el modelo no puede predecir nada util
COLUMNAS_OBLIGATORIAS = ["Subcategoria", "Grupo", "Tipo", "PVP"]

# Valores que se asumen cuando la columna no viene en el archivo
VALORES_POR_DEFECTO = {
    "Marca": "Essence",
    "Tono": "Sin dato",
    "mes_lanzamiento": 7,   # julio: arranque del segundo semestre
}


def estado_datos() -> dict:
    """Compara el calendario del Excel fuente contra el de los datos procesados.

    Permite avisar en pantalla cuando el negocio ya cargo meses nuevos de venta
    POS al archivo pero el modelo todavia corre sobre los datos viejos.

    Returns:
        dict con los meses de cada lado, cuales faltan por procesar y la fecha
        del ultimo recalculo.
    """
    from datetime import datetime

    info = {
        "excel_existe": cfg.ARCHIVO_EXCEL.exists(),
        "procesado_existe": cfg.ARCHIVO_MAESTRO.exists(),
        "meses_excel": [], "meses_procesados": [], "meses_nuevos": [],
        "hay_novedades": False, "error": None, "ultimo_calculo": None,
    }

    if not info["excel_existe"]:
        info["error"] = f"No se encuentra el archivo fuente en {cfg.ARCHIVO_EXCEL}"
        return info

    # Se leen solo los encabezados de una hoja: basta para saber que meses trae
    # y evita cargar un archivo de 50 MB solo para comparar.
    try:
        encabezados = pd.read_excel(cfg.ARCHIVO_EXCEL, sheet_name=cfg.HOJAS_CANAL[0],
                                    nrows=0, engine="openpyxl")
    except PermissionError:
        info["error"] = ("El archivo esta abierto en Excel y no se puede leer. "
                         "Cierralo y vuelve a intentar.")
        return info
    except Exception as e:
        info["error"] = f"No se pudo leer el archivo fuente: {e}"
        return info

    normalizados = ingesta.normalizar_encabezados(encabezados)
    info["meses_excel"] = ingesta.detectar_meses(normalizados)

    if info["procesado_existe"]:
        maestro = pd.read_parquet(cfg.ARCHIVO_MAESTRO)
        info["meses_procesados"] = ingesta.detectar_meses(maestro)
        info["ultimo_calculo"] = datetime.fromtimestamp(
            cfg.ARCHIVO_MAESTRO.stat().st_mtime)

    info["meses_nuevos"] = [m for m in info["meses_excel"]
                            if m not in info["meses_procesados"]]
    info["hay_novedades"] = bool(info["meses_nuevos"])
    return info


def recalcular_todo(con_figuras: bool = False) -> dict:
    """Recalcula el modelo completo leyendo de nuevo el Excel fuente.

    Es lo que ejecuta el boton de la aplicacion cuando el negocio agrega al
    archivo la venta POS de meses nuevos. Rehace toda la cadena: ingesta,
    features, clustering, interpretacion, lanzamientos y el Excel entregable.

    Returns:
        Un resumen de lo que cambio, para mostrarlo en pantalla.
    """
    from . import clustering, evaluation, feature_engineering

    antes = estado_datos()

    utils.titulo("RECALCULO COMPLETO DEL MODELO")
    maestro = ingesta.ejecutar()
    features = feature_engineering.ejecutar(maestro)
    resultado = clustering.ejecutar(features)
    df_final, perfiles = evaluation.ejecutar(resultado)
    modelo, sugeridos = ejecutar(maestro)
    utils.exportar_excel_final(df_final, perfiles, sugeridos)

    if con_figuras:
        from . import eda
        utils.aplicar_estilo_graficas()
        eda.ejecutar()

    return {
        "meses_nuevos": antes["meses_nuevos"],
        "total_meses": len(cfg.MESES),
        "primer_mes": cfg.MESES[0], "ultimo_mes": cfg.MESES[-1],
        "semestre_objetivo": cfg.SEMESTRE_OBJETIVO,
        "registros": len(df_final),
        "clusters": int(df_final.loc[
            df_final["Cluster"] != cfg.CLUSTER_SIN_MOVIMIENTO, "Cluster"].nunique()),
        "lanzamientos_entrenamiento": len(modelo.entrenamiento),
        "lanzamientos_pendientes": len(sugeridos),
        "excel": str(cfg.ARCHIVO_EXCEL_FINAL),
    }


def plantilla_carga() -> pd.DataFrame:
    """Devuelve una plantilla de ejemplo para que el usuario sepa que llenar.

    Se entregan filas de ejemplo reales y no una tabla vacia: es mas rapido
    entender el formato viendo un caso lleno que leyendo una especificacion.
    """
    return pd.DataFrame([
        {"Material": "PB0999001", "Descripcion": "LABIAL ESSENCE ROJO INTENSO 3G",
         "Marca": "Essence", "Subcategoria": "Labios", "Grupo": "Labial",
         "Tipo": "Barra", "Tono": "Rojo", "PVP": 23000, "mes_lanzamiento": 7,
         "Canal": ""},
        {"Material": "PB0999002", "Descripcion": "BRILLO CATRICE NUDE GLOW 10ML",
         "Marca": "Catrice", "Subcategoria": "Labios", "Grupo": "Brillo",
         "Tipo": "Liquido", "Tono": "Nude", "PVP": 18500, "mes_lanzamiento": 7,
         "Canal": ""},
        {"Material": "PB0999003", "Descripcion": "RUBOR ESSENCE ROSADO 5G",
         "Marca": "Essence", "Subcategoria": "Rostro", "Grupo": "Rubor",
         "Tipo": "Polvo", "Tono": "Rosado", "PVP": 21000, "mes_lanzamiento": 11,
         "Canal": "Farmatodo"},
    ])


def detectar_columnas(df: pd.DataFrame) -> tuple[dict, list]:
    """Mapea los encabezados del archivo del usuario a los nombres del modelo.

    Returns:
        (mapa, avisos) donde mapa va de nombre_modelo -> nombre_en_el_archivo.
    """
    normalizados = {utils.normalizar_nombre_columna(c).lower(): c for c in df.columns}
    mapa, avisos = {}, []

    for destino, alias in ALIAS_COLUMNAS.items():
        for nombre in alias:
            if nombre in normalizados:
                mapa[destino] = normalizados[nombre]
                break

    for columna in COLUMNAS_OBLIGATORIAS:
        if columna not in mapa:
            avisos.append(
                f"FALTA la columna obligatoria '{columna}'. "
                f"Encabezados aceptados: {', '.join(ALIAS_COLUMNAS[columna])}"
            )

    for columna, defecto in VALORES_POR_DEFECTO.items():
        if columna not in mapa:
            avisos.append(f"Sin columna '{columna}': se asume '{defecto}' en todas las filas.")

    if "Canal" not in mapa:
        avisos.append(
            "Sin columna 'Canal': se calculara un sugerido para CADA canal "
            f"({', '.join(cfg.HOJAS_CANAL)}), que es como se hace la compra."
        )

    return mapa, avisos


def preparar_carga(df_entrada: pd.DataFrame, modelo: ModeloLanzamiento,
                   canales: list[str] | None = None) -> tuple[pd.DataFrame, list]:
    """Normaliza el archivo cargado y lo deja listo para puntuar.

    Hace tres cosas: renombra columnas, expande por canal si hace falta, y
    resuelve los valores categoricos contra los niveles que el modelo conoce.
    """
    mapa, avisos = detectar_columnas(df_entrada)

    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in mapa]
    if faltantes:
        raise ValueError(
            "El archivo no tiene las columnas obligatorias: "
            + ", ".join(faltantes)
            + ". Descarga la plantilla para ver el formato esperado."
        )

    # --- Renombrado a los nombres internos del modelo ---------------------
    df = pd.DataFrame(index=df_entrada.index)
    for destino, origen in mapa.items():
        df[destino] = df_entrada[origen]

    # Se conserva la fila original del Excel: si algo falla, el usuario necesita
    # saber cual fila de SU archivo revisar, no el indice interno de pandas.
    df["fila_excel"] = df_entrada.index + 2  # +2: encabezado y base 1 de Excel

    # --- Descarte de filas vacias ----------------------------------------
    antes = len(df)
    df = df.dropna(subset=COLUMNAS_OBLIGATORIAS, how="all")
    if len(df) < antes:
        avisos.append(f"Se ignoraron {antes - len(df)} fila(s) completamente vacia(s).")

    # --- Defaults para columnas ausentes ----------------------------------
    for columna, defecto in VALORES_POR_DEFECTO.items():
        if columna not in df.columns:
            df[columna] = defecto
        else:
            df[columna] = df[columna].fillna(defecto)
            # Una celda vacia en el Excel llega como cadena vacia, no como NaN
            df.loc[df[columna].astype(str).str.strip() == "", columna] = defecto

    # --- PVP numerico ------------------------------------------------------
    df["PVP"] = pd.to_numeric(df["PVP"], errors="coerce")
    sin_pvp = df["PVP"].isna() | (df["PVP"] <= 0)
    if sin_pvp.any():
        avisos.append(
            f"{int(sin_pvp.sum())} fila(s) sin PVP valido: se usa la mediana de su "
            f"subcategoria. Filas del Excel: "
            f"{', '.join(map(str, df.loc[sin_pvp, 'fila_excel'].head(10)))}"
        )
        df.loc[sin_pvp, "PVP"] = df.loc[sin_pvp, "Subcategoria"].map(
            modelo.mediana_pvp_subcat).fillna(modelo.pvp_global)

    # --- Mes de lanzamiento -----------------------------------------------
    df["mes_lanzamiento"] = pd.to_numeric(df["mes_lanzamiento"], errors="coerce")
    mes_invalido = ~df["mes_lanzamiento"].between(1, 12)
    if mes_invalido.any():
        avisos.append(f"{int(mes_invalido.sum())} fila(s) con mes de lanzamiento "
                      "invalido: se asume julio (7).")
        df.loc[mes_invalido, "mes_lanzamiento"] = 7
    df["mes_lanzamiento"] = df["mes_lanzamiento"].astype(int)

    # --- Expansion por canal ----------------------------------------------
    canales = canales or cfg.HOJAS_CANAL
    if "Canal" not in df.columns or df["Canal"].astype(str).str.strip().eq("").all():
        # Un producto sin canal definido genera una fila por canal: es la unica
        # forma de que el resultado sea accionable, porque la compra se hace por
        # canal y el canal explica el 47% de la prediccion.
        df = df.drop(columns=[c for c in ["Canal"] if c in df.columns])
        df = df.merge(pd.DataFrame({"Canal": canales}), how="cross")
    else:
        # Archivo mixto: algunas filas con canal y otras sin. Las que no lo traen
        # se expanden a los 3 canales; las que si, se respetan.
        df["Canal"] = df["Canal"].astype(str).str.strip()
        sin_canal = df["Canal"].isin(["", "nan", "None"])
        if sin_canal.any():
            expandidas = (df.loc[sin_canal].drop(columns="Canal")
                            .merge(pd.DataFrame({"Canal": canales}), how="cross"))
            df = pd.concat([df.loc[~sin_canal], expandidas], ignore_index=True)
            avisos.append(f"{int(sin_canal.sum())} fila(s) sin canal: se expandieron "
                          f"a los {len(canales)} canales.")

    # --- Resolucion de valores categoricos --------------------------------
    # Se resuelven contra los niveles que el modelo vio en entrenamiento, para
    # que 'liquido' y 'Líquido' no se traten como categorias distintas.
    # 'Grupo' pasa primero por la unificacion singular/plural, igual que en la
    # ingesta: si no, alguien que escriba 'Pestaninas' no encontraria los
    # comparables de 'Pestanina' y su cohorte caeria a un nivel mas general.
    if "Grupo" in df.columns:
        df["Grupo"] = df["Grupo"].map(utils.unificar_grupo)

    no_reconocidos = {}
    for columna in COLUMNAS_CATEGORICAS:
        if columna not in df.columns:
            continue
        validos = {utils.normalizar_texto(v): v for v in modelo.niveles_validos.get(columna, [])}
        original = df[columna].astype(str)
        resuelto = original.map(lambda v: validos.get(utils.normalizar_texto(v), v))
        desconocidos = sorted(set(original[resuelto == original]) - set(validos.values()))
        if desconocidos:
            no_reconocidos[columna] = desconocidos
        df[columna] = resuelto

    for columna, valores in no_reconocidos.items():
        avisos.append(
            f"Valores no vistos en entrenamiento para '{columna}': "
            f"{', '.join(map(str, valores[:8]))}"
            f"{' ...' if len(valores) > 8 else ''}. "
            "El modelo predice apoyandose en los demas atributos, con menos precision."
        )

    return df.reset_index(drop=True), avisos


def agregar_horizonte_anual(df: pd.DataFrame, X: pd.DataFrame,
                            modelo: ModeloLanzamiento,
                            col_semestral: str) -> pd.DataFrame:
    """Agrega el sugerido a 12 meses y su desagregado mes a mes de rotacion.

    Lo usan los dos caminos que producen sugeridos (la carga de un Excel propio y
    el scoring de los pendientes del archivo fuente), para que ambos entreguen
    exactamente las mismas columnas.

    Args:
        df: filas ya puntuadas en el horizonte semestral.
        X: la matriz de features de esas mismas filas.
        col_semestral: nombre de la columna con el sugerido a 6 meses, que sirve
            de piso para el anual.
    """
    if not modelo.modelos_cuantil_anual:
        return df

    df = df.copy()
    for alpha, pipe in modelo.modelos_cuantil_anual.items():
        df[f"P{int(alpha*100)}_anual"] = np.clip(np.expm1(pipe.predict(X)), 0, None)

    cols_qa = ["P25_anual", "P50_anual", "P75_anual"]
    df[cols_qa] = np.sort(df[cols_qa].to_numpy(), axis=1).round(1)
    df["unidades_sugeridas_anual"] = np.ceil(df["P75_anual"]).astype("int64")

    # Coherencia entre horizontes: el ano nunca puede quedar por debajo del
    # semestre. Son dos modelos entrenados por separado sobre sets distintos
    # (916 y 686 casos), asi que en combinaciones raras pueden cruzarse; sin este
    # piso el Excel mostraria un anual menor que su propio semestre.
    cruces = int((df["unidades_sugeridas_anual"] < df[col_semestral]).sum())
    if cruces:
        utils.log(f"  {cruces} fila(s) con anual < semestral: se ajusta el anual "
                  "al valor semestral (piso de coherencia)")
    df["unidades_sugeridas_anual"] = np.maximum(
        df["unidades_sugeridas_anual"], df[col_semestral])
    df["ingreso_proyectado_anual"] = (
        df["unidades_sugeridas_anual"] * df["PVP"]).round(0)

    # Desagregado mes 1 a mes 12, con la curva real medida en los lanzamientos
    factores = modelo.reparto_mensual_anual
    reparto = np.array([
        _repartir_entero(int(t), factores)
        for t in df["unidades_sugeridas_anual"].to_numpy()
    ])
    for k, columna in enumerate(COLUMNAS_MES):
        df[columna] = reparto[:, k]

    return df


def puntuar_lote(modelo: ModeloLanzamiento, df_preparado: pd.DataFrame) -> pd.DataFrame:
    """Calcula P25/P50/P75, sugerido y respaldo de cohorte para un lote de filas."""
    df = df_preparado.copy()
    X = preparar_features(df, modelo)

    for alpha, pipe in modelo.modelos_cuantil.items():
        df[f"P{int(alpha*100)}"] = np.clip(np.expm1(pipe.predict(X)), 0, None)

    # Los cuantiles se entrenan por separado y pueden cruzarse donde hay pocos
    # datos; se ordenan fila a fila para que el resultado sea siempre coherente.
    columnas_q = ["P25", "P50", "P75"]
    df[columnas_q] = np.sort(df[columnas_q].to_numpy(), axis=1).round(1)

    df["unidades_sugeridas"] = np.ceil(df["P75"]).astype("int64")
    df["ingreso_proyectado"] = (df["unidades_sugeridas"] * df["PVP"]).round(0)

    df = agregar_horizonte_anual(df, X, modelo, col_semestral="unidades_sugeridas")

    # --- Respaldo por cohorte de analogos ---------------------------------
    niveles, n_comp, coh_p75, revisar, ejemplos = [], [], [], [], []
    marca_ok, tono_ok = [], []
    for _, fila in df.iterrows():
        atributos = {c: fila[c] for c in COLUMNAS_CATEGORICAS if c in fila}
        atributos["PVP"] = fila["PVP"]
        comparables, nivel = buscar_cohorte(modelo, atributos)
        p75 = float(comparables["und_primeros_6m"].quantile(0.75))
        diag = diagnostico_cohorte(comparables, nivel, atributos)

        niveles.append(nivel)
        n_comp.append(len(comparables))
        coh_p75.append(round(p75, 1))
        marca_ok.append(diag["marca_respetada"])
        tono_ok.append(diag["tono_respetado"])
        mayor = max(fila["unidades_sugeridas"], p75, 1)
        menor = max(min(fila["unidades_sugeridas"], p75), 1)
        revisar.append(bool(mayor / menor > 2.0))
        # Se guardan 3 comparables concretos: es lo que permite defender la cifra.
        # Ya vienen ordenados por parecido (misma marca y tono primero).
        top = comparables.head(3)
        ejemplos.append(" | ".join(
            f"{str(d)[:30]} [{m}] ({int(u)} und)"
            for d, m, u in zip(top["Descripcion"], top["Marca"], top["und_primeros_6m"])
        ))

    df["cohorte_nivel"] = niveles
    df["cohorte_n"] = n_comp
    df["cohorte_P75"] = coh_p75
    df["marca_respetada"] = marca_ok
    df["tono_respetado"] = tono_ok
    df["requiere_revision"] = revisar
    df["productos_comparables"] = ejemplos

    # Semaforo de confianza: traduce el nivel de cohorte a algo interpretable
    # sin tener que entender la cascada de busqueda.
    #
    # El tono entra en el criterio porque su efecto medido es grande (2-3x dentro
    # del mismo grupo): una estimacion que no pudo conservar el tono merece menos
    # confianza aunque haya coincidido en todo lo demas.
    def confianza(fila):
        if fila["requiere_revision"]:
            return "Baja - revisar"
        if fila["tono_respetado"] and fila["marca_respetada"]:
            return "Alta"
        if fila["tono_respetado"] or "Grupo" in fila["cohorte_nivel"]:
            return "Media"
        return "Baja - revisar"

    df["confianza"] = df.apply(confianza, axis=1)
    return df


def puntuar_archivo(modelo: ModeloLanzamiento,
                    ruta_o_df,
                    hoja=0,
                    canales: list[str] | None = None) -> tuple[pd.DataFrame, list]:
    """Punto de entrada de la carga masiva: Excel/CSV -> sugerido por producto y canal.

    Args:
        modelo: modelo entrenado (`entrenar()`).
        ruta_o_df: ruta a un .xlsx/.csv, un buffer subido, o un DataFrame ya leido.
        hoja: hoja del Excel a leer (nombre o indice).
        canales: canales a calcular. Por defecto los 3 del proyecto.

    Returns:
        (resultados, avisos). `resultados` trae una fila por producto x canal.
    """
    if isinstance(ruta_o_df, pd.DataFrame):
        entrada = ruta_o_df.copy()
    else:
        nombre = str(getattr(ruta_o_df, "name", ruta_o_df))
        if nombre.lower().endswith(".csv"):
            entrada = pd.read_csv(ruta_o_df)
        else:
            entrada = pd.read_excel(ruta_o_df, sheet_name=hoja, engine="openpyxl")

    utils.log(f"Archivo cargado: {len(entrada)} fila(s), {entrada.shape[1]} columna(s)")

    preparado, avisos = preparar_carga(entrada, modelo, canales)
    utils.log(f"Filas a puntuar (producto x canal): {len(preparado)}")

    resultados = puntuar_lote(modelo, preparado)

    columnas = [
        "fila_excel", "Material", "EAN/UPC-Code", "Descripcion", "Marca", "Canal",
        "Subcategoria", "Grupo", "Tipo", "Tono", "PVP", "mes_lanzamiento",
        # Horizonte semestral: la primera compra
        "P25", "P50", "P75", "unidades_sugeridas", "ingreso_proyectado",
        # Horizonte anual: los 12 meses de rotacion
        "P25_anual", "P50_anual", "P75_anual",
        "unidades_sugeridas_anual", "ingreso_proyectado_anual",
        # Desagregado mes a mes de rotacion
        *COLUMNAS_MES,
        # Respaldo y confianza
        "confianza", "cohorte_nivel", "cohorte_n", "cohorte_P75",
        "marca_respetada", "tono_respetado",
        "requiere_revision", "productos_comparables",
    ]
    existentes = [c for c in columnas if c in resultados.columns]
    salida = resultados[existentes].sort_values(
        ["fila_excel", "Canal"] if "fila_excel" in existentes else ["Canal"]
    ).reset_index(drop=True)

    utils.log(f"Unidades sugeridas totales: "
              f"{utils.formatear_miles(salida['unidades_sugeridas'].sum())}")
    utils.log(f"Confianza: {salida['confianza'].value_counts().to_dict()}")

    return salida, avisos


def resumir_lote(resultados: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por producto: una fila por producto con el total y por canal.

    El detalle producto x canal es lo que se pide; este resumen es para la
    conversacion de negocio ("cuanto vale este lanzamiento en total").
    """
    llave = "Material" if "Material" in resultados.columns else "Descripcion"
    if llave not in resultados.columns:
        resultados = resultados.assign(Descripcion=resultados.index.astype(str))
        llave = "Descripcion"

    columnas_id = [c for c in ["Material", "Descripcion", "Marca", "Subcategoria",
                               "Grupo", "Tipo", "Tono", "PVP"]
                   if c in resultados.columns]

    agregados = dict(
        canales=("Canal", "nunique"),
        unidades_sugeridas_total=("unidades_sugeridas", "sum"),
        ingreso_proyectado_total=("ingreso_proyectado", "sum"),
        confianza_minima=("confianza", lambda s: (
            "Baja - revisar" if (s == "Baja - revisar").any()
            else ("Media" if (s == "Media").any() else "Alta"))),
    )
    if "unidades_sugeridas_anual" in resultados.columns:
        agregados["unidades_sugeridas_anual_total"] = ("unidades_sugeridas_anual", "sum")
        agregados["ingreso_proyectado_anual_total"] = ("ingreso_proyectado_anual", "sum")

    base = resultados.groupby(llave, dropna=False).agg(
        **{c: (c, "first") for c in columnas_id if c != llave},
        **agregados,
    ).reset_index()

    # Una columna por canal, para los dos horizontes: el comprador ve el reparto
    # de un vistazo sin tener que sumar filas a mano.
    for valor, prefijo in [("unidades_sugeridas", "sem"),
                           ("unidades_sugeridas_anual", "anio")]:
        if valor not in resultados.columns:
            continue
        pivote = resultados.pivot_table(index=llave, columns="Canal", values=valor,
                                        aggfunc="sum", fill_value=0)
        pivote.columns = [f"{prefijo}_{c}" for c in pivote.columns]
        base = base.merge(pivote.reset_index(), on=llave, how="left")

    # Desagregado mensual del producto: suma de sus canales, mes a mes
    cols_mes = [c for c in COLUMNAS_MES if c in resultados.columns]
    if cols_mes:
        mensual = resultados.groupby(llave, dropna=False)[cols_mes].sum().reset_index()
        base = base.merge(mensual, on=llave, how="left")

    orden = ("unidades_sugeridas_anual_total"
             if "unidades_sugeridas_anual_total" in base.columns
             else "unidades_sugeridas_total")
    return base.sort_values(orden, ascending=False).reset_index(drop=True)


def construir_hoja_mensual(resultados: pd.DataFrame) -> pd.DataFrame:
    """Hoja dedicada al plan mes a mes de rotacion, por producto y canal.

    Es la misma informacion que ya viene en 'Sugerido_por_Canal', pero aislada y
    con una fila de totales: asi el comprador puede pegarla directo en su plan de
    abastecimiento sin tener que esconder veinte columnas de diagnostico.
    """
    cols_mes = [c for c in COLUMNAS_MES if c in resultados.columns]
    if not cols_mes:
        return pd.DataFrame()

    llaves = [c for c in ["Material", "Descripcion", "Marca", "Canal",
                          "Subcategoria", "Grupo", "Tono", "PVP"]
              if c in resultados.columns]

    tabla = resultados[llaves + cols_mes].copy()
    tabla = tabla.rename(columns={c: f"Mes {int(c.split('_')[1])}" for c in cols_mes})
    cols_nuevas = [f"Mes {i+1}" for i in range(len(cols_mes))]

    tabla["Total 12 meses"] = tabla[cols_nuevas].sum(axis=1)
    tabla["Total 6 meses"] = tabla[cols_nuevas[:6]].sum(axis=1)

    # Fila de totales, para leer el volumen agregado de un vistazo
    fila = {c: "" for c in llaves}
    fila[llaves[0]] = "TOTAL"
    for c in cols_nuevas + ["Total 12 meses", "Total 6 meses"]:
        fila[c] = int(tabla[c].sum())

    return pd.concat([tabla.sort_values("Total 12 meses", ascending=False),
                      pd.DataFrame([fila])], ignore_index=True)


def exportar_resultado_lote(resultados: pd.DataFrame, avisos: list,
                            ruta) -> str:
    """Guarda el resultado del lote en un Excel de 4 hojas."""
    resumen = resumir_lote(resultados)
    mensual = construir_hoja_mensual(resultados)
    hoja_avisos = pd.DataFrame({"Revisar": avisos or ["Sin observaciones."]})

    hojas = [("Sugerido_por_Canal", resultados),
             ("Resumen_por_Producto", resumen)]
    if len(mensual):
        hojas.append(("Plan_Mensual_Rotacion", mensual))
    hojas.append(("Validacion", hoja_avisos))

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        for nombre, tabla in hojas:
            tabla.to_excel(writer, sheet_name=nombre, index=False)
            utils.ajustar_ancho_columnas(writer, nombre, tabla)

    utils.log(f"Resultado exportado: {ruta}")
    return str(ruta)


# ---------------------------------------------------------------------------
# 7. IMPORTANCIA DE LOS ATRIBUTOS
# ---------------------------------------------------------------------------


def importancia_atributos(modelo: ModeloLanzamiento, top: int = 15) -> pd.DataFrame:
    """Que atributos explican el volumen de un lanzamiento."""
    pipe = modelo.modelos_cuantil[CUANTIL_SUGERIDO]
    nombres = pipe.named_steps["codificar"].get_feature_names_out()
    importancias = pipe.named_steps["modelo"].feature_importances_

    tabla = (pd.DataFrame({"variable": nombres, "importancia": importancias})
             .sort_values("importancia", ascending=False)
             .head(top).reset_index(drop=True))
    tabla["variable"] = tabla["variable"].str.replace("cat__", "", regex=False) \
                                        .str.replace("remainder__", "", regex=False)
    return tabla


def graficar_importancia(modelo: ModeloLanzamiento, guardar: bool = True):
    """Barras horizontales con la importancia de los atributos."""
    import matplotlib.pyplot as plt

    tabla = importancia_atributos(modelo, top=15).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(tabla["variable"], tabla["importancia"], color="teal")
    ax.set_title("Que atributos explican el volumen de un lanzamiento\n"
                 f"(modelo P{int(CUANTIL_SUGERIDO*100)}, primeros {MESES_HORIZONTE} meses)")
    ax.set_xlabel("Importancia relativa")
    if guardar:
        utils.guardar_figura(fig, "10_importancia_atributos_lanzamiento")
    return fig


def graficar_distribucion_target(modelo: ModeloLanzamiento, guardar: bool = True):
    """Distribucion del volumen de debut historico, por canal."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    ent = modelo.entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].hist(ent["und_primeros_6m"], bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(ent["und_primeros_6m"].median(), color="black", linestyle="--",
                    label=f"Mediana: {ent['und_primeros_6m'].median():.0f} und")
    axes[0].axvline(ent["und_primeros_6m"].quantile(0.75), color="red", linestyle="--",
                    label=f"P75: {ent['und_primeros_6m'].quantile(0.75):.0f} und")
    axes[0].set_title(f"Venta en los primeros {MESES_HORIZONTE} meses\n"
                      f"({len(ent)} lanzamientos historicos)")
    axes[0].set_xlabel("Unidades")
    axes[0].set_ylabel("Lanzamientos")
    axes[0].legend()

    sns.boxplot(data=ent, x="Canal", y="und_primeros_6m", ax=axes[1],
                hue="Canal", palette="Set2", legend=False)
    axes[1].set_yscale("log")
    axes[1].set_title("Volumen de debut por canal (escala log)")
    axes[1].set_ylabel("Unidades en los primeros 6 meses")

    if guardar:
        utils.guardar_figura(fig, "11_distribucion_debut_lanzamientos")
    return fig


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------


def ejecutar(maestro: pd.DataFrame | None = None, guardar: bool = True):
    """Entrena el modelo y puntua los lanzamientos pendientes.

    Returns:
        (modelo, sugeridos) listos para el Excel y el dashboard.
    """
    modelo = entrenar(maestro)
    sugeridos = puntuar_pendientes(modelo, maestro)

    if guardar:
        cfg.crear_directorios()
        sugeridos.to_parquet(cfg.DIR_OUTPUT_DATA / "sugerido_lanzamientos.parquet",
                             index=False)
        modelo.entrenamiento.to_parquet(
            cfg.DIR_PROCESSED / "entrenamiento_lanzamientos.parquet", index=False)
        if modelo.metricas is not None:
            modelo.metricas.to_csv(
                cfg.DIR_OUTPUT_DATA / "metricas_lanzamientos.csv", index=False)
        utils.log(f"Sugeridos guardados en {cfg.DIR_OUTPUT_DATA}")

    return modelo, sugeridos


def _main() -> None:
    """CLI del modulo. Tres modos de uso:

        python -m src.lanzamientos
            Entrena, valida y puntua los lanzamientos pendientes del archivo.

        python -m src.lanzamientos --archivo mis_productos.xlsx
            Puntua un Excel propio con los productos a lanzar.

        python -m src.lanzamientos --plantilla
            Genera la plantilla de carga para llenar.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Sugerido de unidades para productos nuevos (sin historial)")
    parser.add_argument("--archivo", help="Excel o CSV con los productos a lanzar")
    parser.add_argument("--hoja", default=0,
                        help="Hoja del Excel a leer (nombre o indice). Por defecto la primera")
    parser.add_argument("--salida", help="Ruta del Excel de resultados")
    parser.add_argument("--plantilla", action="store_true",
                        help="Genera la plantilla de carga y termina")
    parser.add_argument("--canales", nargs="*", default=None,
                        help="Canales a calcular (por defecto los 3 del proyecto)")
    args = parser.parse_args()

    cfg.crear_directorios()

    # --- Modo plantilla ----------------------------------------------------
    if args.plantilla:
        ruta = cfg.DIR_OUTPUTS / "Plantilla_Carga_Lanzamientos.xlsx"
        plantilla = plantilla_carga()
        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            plantilla.to_excel(writer, sheet_name="Productos", index=False)
            utils.ajustar_ancho_columnas(writer, "Productos", plantilla)
        utils.log(f"Plantilla generada: {ruta}")
        print("\nLlena una fila por producto. Columnas obligatorias: "
              f"{', '.join(COLUMNAS_OBLIGATORIAS)}.")
        print("Deja 'Canal' vacio para recibir un sugerido por cada canal.")
        return

    # --- Modo archivo propio ----------------------------------------------
    if args.archivo:
        modelo = entrenar(validar=False)
        hoja = int(args.hoja) if str(args.hoja).isdigit() else args.hoja

        utils.titulo("PUNTUANDO ARCHIVO DE LANZAMIENTOS")
        resultados, avisos = puntuar_archivo(modelo, args.archivo, hoja=hoja,
                                             canales=args.canales)

        if avisos:
            print("\nOBSERVACIONES DE VALIDACION:")
            for aviso in avisos:
                print(f"  - {aviso}")

        print("\nRESULTADO (una fila por producto x canal):")
        columnas = ["Descripcion", "Canal", "PVP", "P25", "P50",
                    "unidades_sugeridas", "confianza"]
        existentes = [c for c in columnas if c in resultados.columns]
        print(resultados[existentes].to_string(index=False))

        print("\nRESUMEN POR PRODUCTO:")
        resumen = resumir_lote(resultados)
        columnas = [c for c in ["Descripcion", "unidades_sugeridas_total",
                                "confianza_minima"] +
                    [c for c in resumen.columns if c.startswith("sugerido_")]
                    if c in resumen.columns]
        print(resumen[columnas].to_string(index=False))

        ruta = args.salida or (cfg.DIR_OUTPUTS / "Sugerido_Lanzamientos_Cargados.xlsx")
        exportar_resultado_lote(resultados, avisos, ruta)
        return

    # --- Modo por defecto: pipeline completo del modulo --------------------
    modelo, sugeridos = ejecutar()

    # Ejemplo de uso individual: el caso planteado por el negocio
    sugerir_lanzamiento(
        modelo,
        canal="Farmatodo", marca="Essence", subcategoria="Labios",
        grupo="Labial", tipo="Barra", tono="Rojo", pvp=23000,
        mes_lanzamiento=7,
    )


if __name__ == "__main__":
    _main()
