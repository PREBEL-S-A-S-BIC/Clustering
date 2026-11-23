# -*- coding: utf-8 -*-
"""
app.py - Dashboard interactivo del modelo de clustering Cosnova.

Ejecutar desde la raiz del proyecto:
    streamlit run dashboard/app.py

Paginas:
    Inicio                 KPIs generales del modelo
    Analisis de Ventas     evolucion mensual con filtros por canal/marca/subcategoria
    Clusters               scatter PCA interactivo y perfil del cluster elegido
    Recomendacion de Compra tabla filtrable y exportable a Excel

Consulta los resultados generados por el proceso externo de actualización.
"""

from __future__ import annotations

import io
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# El dashboard vive en dashboard/, asi que hay que agregar la raiz al path
# para poder importar el paquete src.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config as cfg  # noqa: E402
from src import utils, ingesta  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIGURACION DE LA PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cosnova | Planeación",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta consistente con las figuras del proyecto
COLORES_CANAL = {"Exito": "#F2C14E", "Farmatodo": "#3E7CB1", "Prosalon": "#8B5FBF"}
COLORES_PRIORIDAD = {
    "Alta": "#2E8B57",
    "Media": "#DAA520",
    "Baja": "#CD5C5C",
    "Sin Movimiento": "#9E9E9E",
}


# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Cargando resultados del modelo ...")
def cargar_datos(version=None):
    """Lee los artefactos del pipeline. Devuelve (clusters, perfiles, metricas)."""
    if not cfg.ARCHIVO_CLUSTERS.exists():
        return None, None, None

    clusters = pd.read_parquet(cfg.ARCHIVO_CLUSTERS)

    ruta_perfiles = cfg.DIR_OUTPUT_DATA / "perfiles_cluster.csv"
    perfiles = pd.read_csv(ruta_perfiles) if ruta_perfiles.exists() else None

    metricas = pd.read_csv(cfg.ARCHIVO_METRICAS) if cfg.ARCHIVO_METRICAS.exists() else None

    return clusters, perfiles, metricas


@st.cache_resource(show_spinner="Preparando sugeridos de lanzamientos ...")
def cargar_modelo_lanzamientos(version=None):
    """Entrena el modelo de lanzamientos una sola vez por sesion.

    Se usa `cache_resource` y no `cache_data` porque lo que se guarda es un objeto
    con modelos de sklearn adentro, no un DataFrame serializable. El entrenamiento
    toma unos segundos, asi que cachearlo evita repetirlo en cada interaccion.
    """
    from src import lanzamientos

    if not cfg.ARCHIVO_MAESTRO.exists():
        return None, None
    # validar=False: la validacion cruzada ya se corrio en el pipeline y aqui solo
    # se necesita el modelo ajustado. Ahorra ~30 segundos de espera al usuario.
    modelo = lanzamientos.entrenar(validar=False)
    return modelo, lanzamientos


def exportar_a_excel(df: pd.DataFrame, nombre_hoja="Datos") -> bytes:
    """Convierte un DataFrame a bytes de Excel para el boton de descarga."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=nombre_hoja, index=False)
    return buffer.getvalue()


def formato_miles(valor) -> str:
    """Formatea numeros grandes para los KPIs."""
    if pd.isna(valor):
        return "-"
    return f"{valor:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# PAGINA 1: INICIO
# ---------------------------------------------------------------------------


def pagina_inicio(clusters: pd.DataFrame, perfiles: pd.DataFrame, metricas: pd.DataFrame):
    st.title("💄 Modelo de Clustering de Productos Cosnova")
    st.caption("ESSENCE y CATRICE | Planeacion de compra semestral por canal de distribucion")

    meses = [m for m in cfg.MESES if m in clusters.columns]
    n_clusters_reales = clusters.loc[clusters["Cluster"] != cfg.CLUSTER_SIN_MOVIMIENTO,
                                    "Cluster"].nunique()
    pct_alta = 100 * (clusters["Prioridad_Compra"] == "Alta").mean()

    # --- KPIs principales -------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registros producto x canal", formato_miles(len(clusters)))
    col2.metric("Productos unicos", formato_miles(clusters["Material"].nunique()))
    col3.metric("Clusters encontrados", n_clusters_reales,
                help="Sin contar el grupo reservado 'Sin Movimiento'")
    col4.metric("Unidades vendidas (30 meses)",
                formato_miles(clusters[meses].to_numpy().sum()))
    col5.metric("% Prioridad Alta", f"{pct_alta:.1f}%")

    st.divider()

    col_izq, col_der = st.columns([1.3, 1])

    with col_izq:
        st.subheader("Distribucion de registros por cluster")
        conteo = (clusters.groupby(["Cluster", "Nombre_Cluster"])
                          .size().reset_index(name="Registros")
                          .sort_values("Cluster"))
        fig = px.bar(conteo, x="Registros", y="Nombre_Cluster", orientation="h",
                     color="Nombre_Cluster", text="Registros")
        fig.update_layout(showlegend=False, height=420,
                          yaxis_title="", xaxis_title="Registros")
        st.plotly_chart(fig, width="stretch")

    with col_der:
        st.subheader("Prioridad de compra")
        prioridad = clusters["Prioridad_Compra"].value_counts().reset_index()
        prioridad.columns = ["Prioridad", "Registros"]
        fig = px.pie(prioridad, names="Prioridad", values="Registros", hole=0.45,
                     color="Prioridad", color_discrete_map=COLORES_PRIORIDAD)
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Participacion por canal")
        por_canal = (clusters.groupby("Canal")[meses].sum().sum(axis=1)
                             .reset_index(name="Unidades"))
        fig = px.bar(por_canal, x="Canal", y="Unidades", color="Canal",
                     color_discrete_map=COLORES_CANAL, text="Unidades")
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Comparacion de algoritmos")
        if metricas is not None:
            st.dataframe(metricas, width="stretch", hide_index=True)
            mejor = metricas.loc[metricas["silhouette"].idxmax(), "modelo"]
            st.success(
                f"Modelo seleccionado: **{mejor}**. Se elige por mayor Silhouette "
                "entre los modelos que clasifican a todos los productos."
            )
        else:
            st.info("Ejecuta el pipeline para ver la comparacion de modelos.")

    with st.expander("Como leer este modelo"):
        st.markdown(
            """
Cada fila del modelo es un **producto en un canal**: el mismo SKU en Exito y en
Farmatodo son dos registros distintos, porque su rotacion y su estacionalidad
tambien lo son.

**Que hace cada cluster.** Los productos se agrupan por como se venden
(volumen, continuidad, tendencia, estacionalidad y variabilidad), no por lo que
son. Eso permite aplicar una misma politica de compra a todo el grupo.

**Prioridad de compra**
- **Alta**: cobertura mayor a 70% y tendencia positiva. Asegurar abastecimiento.
- **Media**: cobertura intermedia o tendencia estable. Compra normal.
- **Baja**: cobertura menor a 40% o caida fuerte. Revisar antes de recomprar.
- **Sin Movimiento**: sin una sola unidad en 30 meses. La cantidad la define el
  area comercial, no el historico.

**Productos sin historial.** Los que nunca vendieron quedan fuera del
clustering (cluster -1): todas sus variables son cero y no hay comportamiento
que agrupar.
            """
        )


# ---------------------------------------------------------------------------
# PAGINA 2: ANALISIS DE VENTAS
# ---------------------------------------------------------------------------


def pagina_ventas(clusters: pd.DataFrame):
    st.title("📈 Analisis de Ventas")
    st.caption("Evolucion del historico de 30 meses (enero 2024 - junio 2026)")

    meses = [m for m in cfg.MESES if m in clusters.columns]

    # --- Filtros -----------------------------------------------------------
    with st.sidebar:
        st.subheader("Filtros")
        canales = st.multiselect("Canal", sorted(clusters["Canal"].unique()),
                                 default=sorted(clusters["Canal"].unique()))
        marcas = st.multiselect("Marca", sorted(clusters["Marca"].unique()),
                                default=sorted(clusters["Marca"].unique()))
        subcategorias = st.multiselect(
            "Subcategoria", sorted(clusters["Subcategoria"].unique()),
            default=sorted(clusters["Subcategoria"].unique()))
        solo_con_venta = st.checkbox("Solo productos con venta", value=True)

    datos = clusters[
        clusters["Canal"].isin(canales)
        & clusters["Marca"].isin(marcas)
        & clusters["Subcategoria"].isin(subcategorias)
    ]
    if solo_con_venta:
        datos = datos[datos["total_unidades"] > 0]

    if datos.empty:
        st.warning("Ningun registro cumple los filtros seleccionados.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", formato_miles(len(datos)))
    col2.metric("Unidades", formato_miles(datos[meses].to_numpy().sum()))
    col3.metric("Ingreso estimado", f"$ {formato_miles(datos['ingreso_estimado'].sum())}")
    col4.metric("Cobertura promedio", f"{datos['cobertura_meses'].mean():.1f}%")

    st.divider()

    # --- Evolucion mensual -------------------------------------------------
    st.subheader("Evolucion mensual de unidades")
    serie = datos[meses].sum().reset_index()
    serie.columns = ["Mes", "Unidades"]
    fig = px.line(serie, x="Mes", y="Unidades", markers=True)
    fig.update_traces(line_color="#3E7CB1", line_width=3)

    # Lineas verticales al cierre de cada semestre: marcan los ciclos de compra.
    # En un eje categorico hay que pasar la POSICION del mes, no su etiqueta:
    # plotly no sabe promediar strings y lanza TypeError si se le da el texto.
    etiquetas = list(serie["Mes"])
    for semestre, meses_sem in cfg.SEMESTRES.items():
        if meses_sem[-1] in etiquetas:
            posicion = etiquetas.index(meses_sem[-1])
            fig.add_vline(x=posicion + 0.5, line_dash="dot", line_color="gray",
                          annotation_text=semestre, annotation_font_size=10)
    fig.update_layout(height=420, xaxis_title="", yaxis_title="Unidades")
    st.plotly_chart(fig, width="stretch")

    # --- Evolucion por canal ----------------------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Evolucion por canal")
        por_canal = datos.groupby("Canal")[meses].sum().T.reset_index()
        por_canal = por_canal.melt(id_vars="index", var_name="Canal",
                                  value_name="Unidades").rename(columns={"index": "Mes"})
        fig = px.line(por_canal, x="Mes", y="Unidades", color="Canal",
                      color_discrete_map=COLORES_CANAL, markers=True)
        fig.update_layout(height=380, xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Unidades por semestre")
        cols_sem = [f"unidades_{s}" for s in cfg.SEMESTRES if f"unidades_{s}" in datos.columns]
        por_semestre = datos[cols_sem].sum().reset_index()
        por_semestre.columns = ["Semestre", "Unidades"]
        por_semestre["Semestre"] = por_semestre["Semestre"].str.replace("unidades_", "")
        fig = px.bar(por_semestre, x="Semestre", y="Unidades", text="Unidades",
                     color="Unidades", color_continuous_scale="Blues")
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Top productos -----------------------------------------------------
    st.subheader("Top 20 productos por unidades vendidas")
    top = datos.nlargest(20, "total_unidades")[
        ["Descripcion", "Marca", "Canal", "Subcategoria", "Grupo",
         "total_unidades", "cobertura_meses", "tendencia", "Nombre_Cluster"]
    ]
    st.dataframe(top, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# PAGINA 3: CLUSTERS
# ---------------------------------------------------------------------------


def pagina_clusters(clusters: pd.DataFrame, perfiles: pd.DataFrame):
    st.title("🎯 Clusters de Productos")
    st.caption("Proyeccion PCA y perfil de cada segmento de comportamiento")

    with st.sidebar:
        st.subheader("Filtros")
        nombres = sorted(clusters["Nombre_Cluster"].unique())
        seleccion = st.multiselect("Clusters a mostrar", nombres, default=nombres)
        opciones_canal = sorted(clusters["Canal"].unique())
        canales = st.multiselect("Canal", opciones_canal, default=opciones_canal)

    datos = clusters[clusters["Nombre_Cluster"].isin(seleccion)
                     & clusters["Canal"].isin(canales)]

    if datos.empty:
        st.warning("Ningun registro cumple los filtros seleccionados.")
        return

    # --- Scatter PCA interactivo ------------------------------------------
    st.subheader("Mapa de clusters en el espacio PCA")
    st.caption("El tamano del punto es el volumen vendido. Pasa el cursor para ver el detalle.")
    fig = px.scatter(
        datos.dropna(subset=["pca_1", "pca_2"]),
        x="pca_1", y="pca_2", color="Nombre_Cluster",
        size="total_unidades", size_max=26, opacity=0.75,
        hover_data={
            "Descripcion": True, "Canal": True, "Marca": True,
            "Subcategoria": True, "total_unidades": ":,.0f",
            "cobertura_meses": ":.1f", "tendencia": ":.1f",
            "Prioridad_Compra": True, "pca_1": False, "pca_2": False,
        },
    )
    fig.update_layout(height=580, legend_title="Cluster",
                      xaxis_title="Componente principal 1",
                      yaxis_title="Componente principal 2")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Clúster asignado por producto y canal")
    st.caption("La asignación se calcula por combinación de material y canal.")
    busqueda = st.text_input("Buscar por código de material o descripción").strip()
    tabla = datos[["Material", "Descripcion", "Canal", "Cluster", "Nombre_Cluster"]]
    if busqueda:
        coincide = tabla["Material"].astype(str).str.contains(
            busqueda, case=False, regex=False, na=False
        ) | tabla["Descripcion"].str.contains(busqueda, case=False, regex=False, na=False)
        tabla = tabla[coincide]
    if tabla.empty:
        st.info("No se encontraron productos con esa búsqueda.")
    else:
        st.dataframe(
            tabla.rename(columns={"Descripcion": "Descripción", "Cluster": "ID de clúster",
                                  "Nombre_Cluster": "Clúster"}),
            width="stretch", hide_index=True, height=400,
        )

    st.divider()

    # --- Perfil del cluster seleccionado ----------------------------------
    st.subheader("Perfil detallado de un cluster")
    elegido = st.selectbox("Selecciona un cluster",
                           sorted(clusters["Nombre_Cluster"].unique()))
    grupo = clusters[clusters["Nombre_Cluster"] == elegido]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registros", formato_miles(len(grupo)))
    col2.metric("Prom. mensual (und)", f"{grupo['promedio_mensual'].mean():.1f}")
    col3.metric("Cobertura", f"{grupo['cobertura_meses'].mean():.1f}%")
    col4.metric("Tendencia", f"{grupo['tendencia'].mean():+.1f} und/mes")
    col5.metric("Variabilidad", f"{grupo['variabilidad'].mean():.2f}")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        composicion = grupo["Subcategoria"].value_counts().reset_index()
        composicion.columns = ["Subcategoria", "Registros"]
        fig = px.pie(composicion, names="Subcategoria", values="Registros", hole=0.4)
        fig.update_layout(title="Por subcategoria", height=330)
        st.plotly_chart(fig, width="stretch")

    with col_b:
        composicion = grupo["Canal"].value_counts().reset_index()
        composicion.columns = ["Canal", "Registros"]
        fig = px.pie(composicion, names="Canal", values="Registros", hole=0.4,
                     color="Canal", color_discrete_map=COLORES_CANAL)
        fig.update_layout(title="Por canal", height=330)
        st.plotly_chart(fig, width="stretch")

    with col_c:
        composicion = grupo["Marca"].value_counts().reset_index()
        composicion.columns = ["Marca", "Registros"]
        fig = px.pie(composicion, names="Marca", values="Registros", hole=0.4)
        fig.update_layout(title="Por marca", height=330)
        st.plotly_chart(fig, width="stretch")

    # --- Patron semestral del cluster -------------------------------------
    st.subheader("Patron de venta semestral del cluster")
    cols_sem = [f"unidades_{s}" for s in cfg.SEMESTRES if f"unidades_{s}" in grupo.columns]
    patron = grupo[cols_sem].mean().reset_index()
    patron.columns = ["Semestre", "Promedio de unidades"]
    patron["Semestre"] = patron["Semestre"].str.replace("unidades_", "")
    fig = px.bar(patron, x="Semestre", y="Promedio de unidades", text_auto=".0f",
                 color="Promedio de unidades", color_continuous_scale="Teal")
    fig.update_layout(height=330, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    # --- Tabla de perfiles completa ---------------------------------------
    if perfiles is not None:
        with st.expander("Tabla comparativa de todos los clusters"):
            columnas = ["Cluster", "Nombre_Cluster", "n_registros",
                        "moda_Subcategoria", "moda_Marca", "moda_Canal",
                        "promedio_mensual_prom", "cobertura_meses_prom",
                        "tendencia_prom", "variabilidad_prom"]
            existentes = [c for c in columnas if c in perfiles.columns]
            st.dataframe(perfiles[existentes], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# PAGINA 4: RECOMENDACION DE COMPRA
# ---------------------------------------------------------------------------


def pagina_recomendacion(clusters: pd.DataFrame):
    st.title("🛒 Recomendacion de Compra")
    st.caption(f"Proyeccion sugerida para {cfg.SEMESTRE_OBJETIVO} por producto y canal")

    with st.sidebar:
        st.subheader("Filtros")
        canales = st.multiselect("Canal", sorted(clusters["Canal"].unique()),
                                 default=sorted(clusters["Canal"].unique()))
        prioridades = st.multiselect(
            "Prioridad de compra", ["Alta", "Media", "Baja", "Sin Movimiento"],
            default=["Alta", "Media", "Baja"])
        nombres = sorted(clusters["Nombre_Cluster"].unique())
        seleccion = st.multiselect("Cluster", nombres, default=nombres)
        marcas = st.multiselect("Marca", sorted(clusters["Marca"].unique()),
                                default=sorted(clusters["Marca"].unique()))
        minimo = st.number_input("Unidades sugeridas minimas", min_value=0, value=0, step=10)

    datos = clusters[
        clusters["Canal"].isin(canales)
        & clusters["Prioridad_Compra"].isin(prioridades)
        & clusters["Nombre_Cluster"].isin(seleccion)
        & clusters["Marca"].isin(marcas)
        & (clusters["unidades_sugeridas_semestre"] >= minimo)
    ]

    if datos.empty:
        st.warning("Ningun registro cumple los filtros seleccionados.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros a comprar", formato_miles(len(datos)))
    col2.metric(f"Unidades sugeridas {cfg.SEMESTRE_OBJETIVO}",
                formato_miles(datos["unidades_sugeridas_semestre"].sum()))
    col3.metric("Ingreso proyectado",
                f"$ {formato_miles(datos['ingreso_proyectado_semestre'].sum())}")
    col4.metric("Venta real S1 2026", formato_miles(datos["unidades_S1_2026"].sum()))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Unidades sugeridas por canal y prioridad")
        resumen = (datos.groupby(["Canal", "Prioridad_Compra"])
                        ["unidades_sugeridas_semestre"].sum().reset_index())
        fig = px.bar(resumen, x="Canal", y="unidades_sugeridas_semestre",
                     color="Prioridad_Compra", color_discrete_map=COLORES_PRIORIDAD,
                     barmode="group", text_auto=".2s")
        fig.update_layout(height=380, yaxis_title="Unidades sugeridas")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Metodo usado en la proyeccion")
        metodos = datos["metodo_proyeccion"].value_counts().reset_index()
        metodos.columns = ["Metodo", "Registros"]
        fig = px.bar(metodos, x="Registros", y="Metodo", orientation="h",
                     text="Registros", color="Registros",
                     color_continuous_scale="Blues")
        fig.update_layout(height=380, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Tabla detallada ---------------------------------------------------
    st.subheader("Detalle de la recomendacion")
    columnas = [
        "Material", "Descripcion", "Canal", "Cluster", "Nombre_Cluster",
        "Marca", "Subcategoria", "Grupo", "PVP", "Prioridad_Compra",
        "unidades_S2_2024", "unidades_S2_2025", "unidades_S1_2026",
        "factor_cluster_S2", "unidades_sugeridas_semestre",
        "ingreso_proyectado_semestre", "metodo_proyeccion",
    ]
    existentes = [c for c in columnas if c in datos.columns]
    tabla = datos[existentes].sort_values("unidades_sugeridas_semestre", ascending=False)

    st.dataframe(tabla, width="stretch", hide_index=True, height=430)

    st.download_button(
        label="⬇️ Descargar recomendacion filtrada en Excel",
        data=exportar_a_excel(tabla, "Recomendacion_Compra"),
        file_name=f"Recomendacion_Compra_{cfg.SEMESTRE_OBJETIVO}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("Como se calcula la proyeccion"):
        st.markdown(
            f"""
La cantidad sugerida para **{cfg.SEMESTRE_OBJETIVO}** se estima con el primer
metodo que aplique a cada producto:

1. **S1_2026 x factor cluster** — el producto vendio en el semestre mas
   reciente. Se toma esa venta real y se ajusta por el factor S1 a S2 de su
   cluster (calibrado con 2025). Es el metodo mas confiable porque parte de un
   dato observado y actual.
2. **Promedio S2 propio** — no vendio en S1 2026 pero tiene historia en
   segundos semestres anteriores. Se usa el promedio de sus propios S2.
3. **Promedio S2 del cluster** — producto con venta pero sin ningun S2 propio
   (lanzamiento reciente). Se usa el promedio de S2 de los productos activos de
   su cluster. Este es el caso que justifica el modelo: el grupo aporta la
   referencia que el producto todavia no tiene.
4. **Sin historial** — nunca vendio. El modelo no proyecta; la cantidad la debe
   definir el area comercial segun el plan de lanzamiento.

El resultado se **redondea hacia arriba**: en una compra semestral un excedente
marginal cuesta menos que un quiebre de stock, que se lleva la venta completa.
El factor de cluster se acota entre 0,5x y 2,0x para que un grupo con base
pequena no dispare la recomendacion.
            """
        )


# ---------------------------------------------------------------------------
# PAGINA 5: LANZAMIENTO NUEVO
# ---------------------------------------------------------------------------


def pagina_lanzamiento(clusters: pd.DataFrame):
    st.title("🚀 Sugerido para Productos Nuevos")
    st.caption("Estima unidades a pedir para productos sin historial, "
               "a partir de sus atributos")

    version = cfg.ARCHIVO_MAESTRO.stat().st_mtime_ns if cfg.ARCHIVO_MAESTRO.exists() else None
    modelo, modulo = cargar_modelo_lanzamientos(version)
    if modelo is None:
        st.info("Solicita al responsable de la información que prepare los "
                "datos de lanzamientos.")
        return

    ent = modelo.entrenamiento

    st.markdown(
        f"""
El modelo aprendió de **{len(ent)} lanzamientos históricos reales** con al menos
6 meses de venta observables.
Estima cuántas unidades vende un producto nuevo en sus **primeros 6 meses**, que es
justo lo que se pide en la primera compra semestral.
        """
    )

    tab_lote, tab_individual, tab_pendientes = st.tabs([
        "📄 Cargar Excel (varios productos)",
        "🔍 Consultar un producto",
        "📋 Lanzamientos pendientes del archivo",
    ])

    with tab_lote:
        _tab_carga_excel(modelo, modulo)
    with tab_individual:
        _tab_producto_individual(modelo, modulo)
    with tab_pendientes:
        _tab_pendientes()

    with st.expander("Cómo funciona y qué tan confiable es"):
        _explicacion_modelo()


def _tab_carga_excel(modelo, modulo):
    """Carga masiva: el usuario sube un Excel con los productos a lanzar."""
    st.subheader("1. Prepara tu archivo")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"""
Una fila por producto. Columnas **obligatorias**:
`Subcategoria`, `Grupo`, `Tipo`, `PVP`.

Opcionales: `Material`, `Descripcion`, `Marca`, `Tono`, `mes_lanzamiento`, `Canal`.

**Deja `Canal` vacío** y recibes un sugerido para cada uno de los
{len(cfg.HOJAS_CANAL)} canales — que es como se hace la compra. Si pones un canal
específico, solo se calcula ese.

Los encabezados se detectan solos: sirve `Subcategoría`, `subcategoria` o `Linea`;
`PVP`, `Precio` o `Precio de venta`. Tampoco importan mayúsculas ni tildes en los
valores (`liquido` y `Líquido` son lo mismo).
            """
        )
    with col_b:
        plantilla = modulo.plantilla_carga()
        st.download_button(
            "⬇️ Descargar plantilla",
            data=exportar_a_excel(plantilla, "Productos"),
            file_name="Plantilla_Carga_Lanzamientos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        with st.expander("Valores válidos"):
            for columna in ["Subcategoria", "Grupo", "Tipo", "Tono", "Marca"]:
                valores = modelo.niveles_validos.get(columna, [])
                st.markdown(f"**{columna}** ({len(valores)})")
                st.caption(", ".join(map(str, valores)))

    st.divider()
    st.subheader("2. Sube el archivo")

    archivo = st.file_uploader("Excel o CSV con los productos a lanzar",
                               type=["xlsx", "xlsm", "csv"])
    if archivo is None:
        st.info("Sube un archivo para calcular los sugeridos. "
                "Si no tienes uno, descarga la plantilla y llénala.")
        return

    # Si es Excel con varias hojas, hay que dejar elegir cuál
    hoja = 0
    if not archivo.name.lower().endswith(".csv"):
        try:
            hojas = pd.ExcelFile(archivo).sheet_names
            if len(hojas) > 1:
                hoja = st.selectbox("¿Qué hoja contiene los productos?", hojas)
            else:
                hoja = hojas[0]
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            return

    canales = st.multiselect("Canales a calcular", cfg.HOJAS_CANAL,
                             default=cfg.HOJAS_CANAL)
    if not canales:
        st.warning("Selecciona al menos un canal.")
        return

    try:
        resultados, avisos = modulo.puntuar_archivo(modelo, archivo, hoja=hoja,
                                                    canales=canales)
    except ValueError as e:
        # Error esperable: al archivo le faltan columnas obligatorias
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error al procesar el archivo: {type(e).__name__}: {e}")
        return

    # --- Validación --------------------------------------------------------
    if avisos:
        with st.expander(f"⚠️ {len(avisos)} observación(es) de validación", expanded=True):
            for aviso in avisos:
                if aviso.startswith("FALTA"):
                    st.error(aviso)
                else:
                    st.warning(aviso)

    st.divider()
    st.subheader("3. Resultado")

    resumen = modulo.resumir_lote(resultados)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Productos", formato_miles(len(resumen)))
    col2.metric("Unidades · primeros 6 meses",
                formato_miles(resultados["unidades_sugeridas"].sum()))
    if "unidades_sugeridas_anual" in resultados.columns:
        col3.metric("Unidades · 12 meses",
                    formato_miles(resultados["unidades_sugeridas_anual"].sum()))
    else:
        col3.metric("Filas producto × canal", formato_miles(len(resultados)))
    col4.metric("Ingreso proyectado",
                f"$ {formato_miles(resultados['ingreso_proyectado'].sum())}")

    _simulador_mensual(modelo, resultados, "lote")

    # --- Semáforo de confianza --------------------------------------------
    conteo = resultados["confianza"].value_counts()
    n_revisar = int(conteo.get("Baja - revisar", 0))
    if n_revisar:
        st.warning(
            f"**{n_revisar} de {len(resultados)} filas tienen confianza baja** y "
            "conviene validarlas con el área comercial antes de pedir. Son "
            "combinaciones de atributos con poco respaldo histórico: mira la columna "
            "`cohorte_nivel` para ver qué tan específica pudo ser la búsqueda."
        )

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        orden = ["Alta", "Media", "Baja - revisar"]
        datos = (resultados["confianza"].value_counts()
                 .reindex(orden).fillna(0).reset_index())
        datos.columns = ["Confianza", "Filas"]
        fig = px.pie(datos, names="Confianza", values="Filas", hole=0.45,
                     color="Confianza",
                     color_discrete_map={"Alta": "#2E8B57", "Media": "#DAA520",
                                         "Baja - revisar": "#CD5C5C"})
        fig.update_layout(height=330, title="Confianza de las estimaciones")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        por_canal = (resultados.groupby("Canal")["unidades_sugeridas"]
                     .sum().reset_index())
        fig = px.bar(por_canal, x="Canal", y="unidades_sugeridas", color="Canal",
                     color_discrete_map=COLORES_CANAL, text_auto=".2s")
        fig.update_layout(height=330, showlegend=False,
                          title="Unidades sugeridas por canal",
                          yaxis_title="Unidades")
        st.plotly_chart(fig, width="stretch")

    # --- Tablas ------------------------------------------------------------
    st.markdown("**Resumen por producto** (total y reparto por canal)")
    st.dataframe(resumen, width="stretch", hide_index=True)

    st.markdown("**Detalle por producto × canal**")
    solo_revisar = st.checkbox("Mostrar solo las filas de confianza baja", value=False)
    vista = resultados[resultados["confianza"] == "Baja - revisar"] if solo_revisar \
        else resultados
    st.dataframe(vista, width="stretch", hide_index=True, height=400)

    # --- Descarga ----------------------------------------------------------
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        resultados.to_excel(writer, sheet_name="Sugerido_por_Canal", index=False)
        resumen.to_excel(writer, sheet_name="Resumen_por_Producto", index=False)
        mensual = modulo.construir_hoja_mensual(resultados)
        if not mensual.empty:
            mensual.to_excel(writer, sheet_name="Plan_Mensual_Rotacion", index=False)
        pd.DataFrame({"Revisar": avisos or ["Sin observaciones."]}).to_excel(
            writer, sheet_name="Validacion", index=False)

    st.download_button(
        "⬇️ Descargar resultados y plan mensual",
        data=buffer.getvalue(),
        file_name="Sugerido_Lanzamientos_Cargados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def _simulador_mensual(modelo, resultados, clave):
    """Escenario editable para los productos y canales elegidos."""
    from src import simulacion

    firma = hashlib.sha256(resultados.to_json(orient="split").encode()).hexdigest()[:16]
    prefijo = f"sim_{clave}_{firma}"
    datos = resultados.reset_index(drop=True)
    with st.expander("Simular ajustes mensuales", expanded=False):
        st.caption(
            "Los pesos estacionales redistribuyen el total del modelo entre meses. "
            "El porcentaje comercial se aplica después y modifica el volumen."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            opciones = [6, 12] if "unidades_sugeridas_anual" in datos else [6]
            horizonte = st.selectbox("Horizonte en meses", opciones, key=f"{prefijo}_horizonte")
        with c2:
            hoy = datetime.now(timezone(timedelta(hours=-5)))
            anio = st.number_input("Año de lanzamiento", min_value=1, max_value=9998,
                                   value=hoy.year, step=1, key=f"{prefijo}_anio")
        with c3:
            opciones_canal = sorted(datos["Canal"].unique())
            canales = st.multiselect("Canales del escenario", opciones_canal,
                                      default=opciones_canal, key=f"{prefijo}_canales")
        elegibles = datos.index[datos["Canal"].isin(canales)].tolist()
        if st.checkbox("Elegir productos específicos", key=f"{prefijo}_elegir"):
            elegidos = st.multiselect(
                "Productos y canales", elegibles,
                format_func=lambda i: (
                    f"{i + 1} · {datos.loc[i].get('Descripcion', 'Producto')} "
                    f"· {datos.loc[i, 'Canal']}"
                ),
                key=f"{prefijo}_productos",
            )
        else:
            elegidos = elegibles
        if not elegidos:
            st.info("Selecciona al menos un producto y canal para simular.")
            return
        seleccion = datos.loc[elegidos]
        plan = simulacion.construir_plan(seleccion, modelo, horizonte, anio)
        configuracion = pd.DataFrame({
            "Mes": sorted(plan["Mes"].unique()),
            "Peso estacional": 1.0,
            "Ajuste comercial (%)": 0.0,
            "Motivo": "",
        })
        editor_firma = hashlib.sha256(plan.to_json(orient="split").encode()).hexdigest()[:16]
        editor = f"{prefijo}_{editor_firma}"
        revision = st.session_state.get(f"{editor}_revision", 0)
        st.caption(
            "El mes inicial de cada producto corresponde a su mes de lanzamiento. "
            "Un peso 2 duplica la importancia relativa de ese mes frente a un peso 1. "
            "Los ajustes se aplican a los productos y canales seleccionados. "
            "Cambiar el alcance, el año o el horizonte inicia otro escenario."
        )
        if st.button("Restablecer ajustes", key=f"{editor}_restablecer"):
            st.session_state[f"{editor}_revision"] = revision + 1
            st.rerun()
        ajustes = st.data_editor(
            configuracion, hide_index=True, num_rows="fixed", width="stretch",
            disabled=["Mes"], key=f"{editor}_tabla_{revision}",
            column_config={
                "Mes": st.column_config.TextColumn("Mes calendario"),
                "Peso estacional": st.column_config.NumberColumn(
                    min_value=0, step=0.1, format="%.2f", required=True,
                ),
                "Ajuste comercial (%)": st.column_config.NumberColumn(
                    min_value=-100, step=1, format="%.1f", required=True,
                    help="Porcentaje aplicado al volumen después de redistribuirlo.",
                ),
                "Motivo": st.column_config.TextColumn(
                    help="Promoción, exhibición, negociación u otra razón del ajuste.",
                ),
            },
        )
        try:
            escenario = simulacion.aplicar_ajustes(plan, ajustes)
        except ValueError as error:
            st.error(str(error))
            return
        base = int(escenario["Base del modelo"].sum())
        ajustado = int(escenario["Unidades ajustadas"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total del modelo", formato_miles(base))
        m2.metric("Total ajustado", formato_miles(ajustado))
        m3.metric("Diferencia en unidades", formato_miles(ajustado - base))
        columnas_volumen = ["Base del modelo", "Con estacionalidad", "Unidades ajustadas"]
        mensual = escenario.groupby("Mes", as_index=False)[columnas_volumen].sum()
        grafica = mensual.melt(id_vars="Mes", var_name="Escenario", value_name="Unidades")
        fig = px.line(grafica, x="Mes", y="Unidades", color="Escenario", markers=True)
        fig.update_layout(height=340, xaxis_title="Mes calendario", yaxis_title="Unidades")
        st.plotly_chart(fig, width="stretch", key=f"{prefijo}_grafica")
        detalle = escenario.drop(columns="Factor base")
        st.dataframe(detalle, hide_index=True, width="stretch")
        st.caption(
            "El redondeo se realiza por producto y canal, conservando su total ajustado. "
            "Descarga el resultado para conservar los porcentajes y sus motivos."
        )
        identificacion = [c for c in ["Registro", "Material", "Descripcion", "Marca",
                                     "Canal", "Subcategoria", "Grupo", "Tipo", "Tono", "PVP"]
                          if c in detalle]
        resumen = detalle.groupby(identificacion, dropna=False, as_index=False)[
            columnas_volumen + ["Diferencia"]
        ].sum()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            detalle.to_excel(writer, sheet_name="Simulacion_Mensual", index=False)
            ajustes.to_excel(writer, sheet_name="Ajustes", index=False)
            resumen.to_excel(writer, sheet_name="Resumen", index=False)
        st.download_button(
            "Descargar simulación en Excel", data=buffer.getvalue(),
            file_name="Simulacion_Lanzamientos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{prefijo}_descarga",
        )


def _tab_pendientes():
    """Los lanzamientos pendientes que ya venían en el archivo fuente."""
    ruta_sug = cfg.DIR_OUTPUT_DATA / "sugerido_lanzamientos.parquet"
    if not ruta_sug.exists():
        st.info("Ejecuta `python run_pipeline.py` para calcular los pendientes "
                "del archivo fuente.")
        return

    sug = pd.read_parquet(ruta_sug)
    st.caption(
        f"{len(sug)} registros del archivo fuente sin una sola venta en 30 meses "
        "(en su mayoría novedades 2026). El modelo ya les puso cifra."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", formato_miles(len(sug)))
    c2.metric("Unidades sugeridas",
              formato_miles(sug["unidades_sugeridas_lanzamiento"].sum()))
    c3.metric("Requieren revisión",
              f"{int(sug['requiere_revision'].sum())} "
              f"({100 * sug['requiere_revision'].mean():.0f}%)")

    filtro_canal = st.multiselect("Filtrar por canal", sorted(sug["Canal"].unique()),
                                  default=sorted(sug["Canal"].unique()),
                                  key="pend_canal")
    solo_revisar = st.checkbox("Solo los que requieren revisión manual", value=False,
                              key="pend_rev")
    vista = sug[sug["Canal"].isin(filtro_canal)]
    if solo_revisar:
        vista = vista[vista["requiere_revision"]]

    st.dataframe(vista, width="stretch", hide_index=True, height=380)
    st.download_button(
        "⬇️ Descargar sugerido de pendientes en Excel",
        data=exportar_a_excel(vista, "Sugerido_Lanzamientos"),
        file_name="Sugerido_Lanzamientos_Pendientes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _tab_producto_individual(modelo, modulo):
    """Consulta de un solo producto mediante formulario."""
    ent = modelo.entrenamiento

    # --- Formulario de atributos ------------------------------------------
    st.subheader("Atributos del producto a lanzar")

    # Unico filtro en cascada: Subcategoria limita Grupo.
    # Tipo y Tono quedan completos a proposito: un producto nuevo puede ser la
    # primera combinacion de su tipo, y restringirlos bloquearia justamente los
    # lanzamientos mas novedosos.
    col1, col2, col3 = st.columns(3)
    with col1:
        canales = modulo.opciones_validas(modelo, "Canal")
        canal = st.selectbox("Canal", canales,
                             index=canales.index("Farmatodo") if "Farmatodo" in canales else 0)
        marca = st.selectbox("Marca", modulo.opciones_validas(modelo, "Marca"))
        subcategoria = st.selectbox("Subcategoría",
                                    modulo.opciones_validas(modelo, "Subcategoria"))
    with col2:
        grupo = st.selectbox("Grupo", modulo.opciones_validas(
            modelo, "Grupo", {"Subcategoria": subcategoria}))
        tipo = st.selectbox("Tipo", modulo.opciones_validas(modelo, "Tipo"))
        tono = st.selectbox(
            "Tono", modulo.opciones_validas(modelo, "Tono"),
            help="Dentro de un mismo grupo la venta varía entre 2 y 3 veces "
                 "según el tono.")
        sel = {"Marca": marca, "Subcategoria": subcategoria, "Grupo": grupo,
               "Tipo": tipo, "Tono": tono}
    with col3:
        pvp = st.number_input("PVP (pesos)", min_value=1000, max_value=200000,
                              value=int(modulo.pvp_tipico(modelo, sel)), step=500,
                              help="Se propone el PVP típico de productos similares.")
        mes = st.selectbox(
            "Mes de lanzamiento",
            list(range(1, 13)), index=6,
            format_func=lambda m: ["Enero", "Febrero", "Marzo", "Abril", "Mayo",
                                   "Junio", "Julio", "Agosto", "Septiembre",
                                   "Octubre", "Noviembre", "Diciembre"][m - 1],
        )
        st.caption("Julio a diciembre = segundo semestre, que históricamente vende más.")

    firma = (canal, marca, subcategoria, grupo, tipo, tono, pvp, mes,
             cfg.ARCHIVO_MAESTRO.stat().st_mtime_ns)
    if st.button("Calcular sugerido", type="primary"):
        calculo = modulo.sugerir_lanzamiento(
            modelo, canal=canal, marca=marca, subcategoria=subcategoria,
            grupo=grupo, tipo=tipo, tono=tono, pvp=pvp,
            mes_lanzamiento=mes, mostrar=False,
        )
        st.session_state["consulta_lanzamiento"] = (firma, calculo)
    guardado = st.session_state.get("consulta_lanzamiento")
    if guardado is None or guardado[0] != firma:
        st.info("Define los atributos y presiona **Calcular sugerido**.")
        return
    r = guardado[1]

    st.divider()

    # --- Resultado ---------------------------------------------------------
    st.subheader("Sugerido de compra")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Escenario bajo (P25)", formato_miles(r["P25"]), help="25% de los comparables vendió menos")
    col2.metric("Escenario medio (P50)", formato_miles(r["P50"]), help="La mitad vendió menos")
    col3.metric("SUGERIDO 6 MESES (P75)", formato_miles(r["unidades_sugeridas"]),
                help="Cubre el 75% de los escenarios: nivel de servicio elegido por el negocio")
    col4.metric("Ingreso proyectado", f"$ {formato_miles(r['ingreso_proyectado'])}")

    base_simulacion = pd.DataFrame([{
        **r["atributos"], "unidades_sugeridas": r["unidades_sugeridas"],
    }])
    base_simulacion = modulo.agregar_horizonte_anual(
        base_simulacion, modulo.preparar_features(base_simulacion, modelo),
        modelo, col_semestral="unidades_sugeridas",
    )
    _simulador_mensual(modelo, base_simulacion, "individual")

    # --- Reparto mensual ---------------------------------------------------
    st.markdown("**Cómo repartir esa cantidad mes a mes**")
    tabla_mes = modulo.sugerido_por_mes(modelo, r["unidades_sugeridas"])
    cm1, cm2 = st.columns([1, 2])
    with cm1:
        st.metric("Promedio mensual", formato_miles(r["unidades_sugeridas"] / 6),
                  help="Total ÷ 6. El reparto real usa la curva medida de los "
                       "lanzamientos, con el primer mes ya corregido por el "
                       "sesgo de los productos que entran a mitad de mes.")
    with cm2:
        st.dataframe(tabla_mes, width="stretch", hide_index=True)

    # --- Gráfica de escenarios contra los comparables reales --------------
    comparables = r["comparables"]
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=comparables["und_primeros_6m"], name="Comparables históricos",
        boxpoints="all", jitter=0.4, pointpos=-1.6, marker_color="#3E7CB1",
        text=comparables["Descripcion"], hovertemplate="%{text}<br>%{y} unidades<extra></extra>",
    ))
    for etiqueta, valor, color in [("P25", r["P25"], "#CD5C5C"),
                                   ("P50", r["P50"], "#DAA520"),
                                   ("P75 (sugerido)", r["unidades_sugeridas"], "#2E8B57")]:
        fig.add_hline(y=valor, line_dash="dash", line_color=color,
                      annotation_text=f"{etiqueta}: {valor:,.0f}".replace(",", "."),
                      annotation_position="right")
    fig.update_layout(
        height=430, showlegend=False,
        title="El sugerido frente a la venta real de los productos comparables",
        yaxis_title="Unidades en los primeros 6 meses",
    )
    st.plotly_chart(fig, width="stretch")

    # --- Respaldo y señal de confianza ------------------------------------
    st.subheader("De dónde sale el número")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
**Criterio de búsqueda alcanzado:** `{r['cohorte_nivel']}`

**Productos comparables encontrados:** {r['cohorte_n']}

**Su venta real en los primeros 6 meses:**
- P25: {formato_miles(r['cohorte_P25'])} unidades
- P50: {formato_miles(r['cohorte_P50'])} unidades
- P75: {formato_miles(r['cohorte_P75'])} unidades

**Marca {marca}:** {'coincide en todos' if r['marca_respetada'] else f"solo {r['pct_misma_marca']}%"}

**Tono {tono}:** {'coincide en todos' if r['tono_respetado'] else f"solo {r['pct_mismo_tono']}%"}
        """)
    with col_b:
        if r["requiere_revision"]:
            st.warning(
                f"**Revisar antes de pedir.** El modelo sugiere "
                f"{formato_miles(r['unidades_sugeridas'])} unidades y los productos "
                f"comparables apuntan a {formato_miles(r['cohorte_sugerido'])} "
                f"(diferencia de {r['divergencia']}x).\n\n"
                "Esta combinación de atributos tiene poco respaldo histórico. "
                "Validar con el área comercial."
            )
        else:
            st.success(
                f"**Estimación respaldada.** Modelo y comparables coinciden "
                f"(diferencia de solo {r['divergencia']}x). "
                "El número tiene evidencia histórica detrás."
            )

    st.markdown("**Productos comparables y su venta real:**")
    tabla = comparables[["Descripcion", "Marca", "Canal", "Grupo", "Tipo", "Tono",
                         "PVP", "debut_mes", "und_primeros_6m"]] \
        .sort_values("und_primeros_6m", ascending=False) \
        .rename(columns={"und_primeros_6m": "Vendió en sus primeros 6 meses",
                         "debut_mes": "Debutó en"})
    st.dataframe(tabla, width="stretch", hide_index=True, height=280)


def _explicacion_modelo():
    """Explicacion compartida por las tres pestanas."""
    st.markdown(
            """
**Dos motores que se cruzan entre sí**

1. **Modelo cuantílico** (Gradient Boosting): entrega P25, P50 y P75 directamente.
   Es la cifra oficial del sugerido.
2. **Cohorte de análogos**: busca productos históricos parecidos en cascada, de lo
   más específico a lo más general — `Canal + Grupo + Tipo + Tono + PVP ±15%`,
   luego sin tono, luego sin PVP, y así hasta quedarse solo con el canal. Se detiene
   en cuanto reúne 8 comparables. **El criterio que alcanzó es la medida de
   confianza:** llegar a `Canal + Grupo + Tipo + Tono` es evidencia fuerte; quedarse
   en `Canal` a secas significa que no había nada parecido.

Si ambos motores difieren por más del doble, el registro se marca para revisión.

**Desempeño medido** (validación cruzada agrupada por Material, 5 folds):

| Método | MAE | MAE_log | Dentro de factor 2 |
|---|---|---|---|
| Mediana por canal (baseline) | 312 | 0,916 | 44,2% |
| **Modelo de atributos** | **227** | **0,628** | **63,9%** |

El modelo mejora **31%** sobre lo que haría un analista sin modelo. Los cuantiles
están calibrados: el P75 cubre el 71,8% real de los casos.

**Qué atributos pesan más**

El canal domina (47% de la importancia): un lanzamiento en Farmatodo vende ~10x
lo que el mismo producto en Prosalon. Le siguen el PVP relativo a su subcategoría,
el mes de lanzamiento, la marca y luego grupo, tipo y tono.

**Límites honestos**

- Es una estimación con **incertidumbre real**: acierta dentro del factor 2 en el
  64% de los casos, no es una cifra exacta. Por eso se entrega el rango.
- No conoce el plan de medios, la exhibición negociada ni el respaldo publicitario.
  Dos productos idénticos con distinto apoyo comercial venden muy distinto y el
  modelo no puede verlo.
- Un tono o grupo nunca visto se predice apoyándose en el resto de los atributos,
  con menos precisión. La señal de revisión avisa cuando eso pasa.
            """
        )


# ---------------------------------------------------------------------------
# NAVEGACION
# ---------------------------------------------------------------------------


def main():
    version = cfg.ARCHIVO_CLUSTERS.stat().st_mtime_ns if cfg.ARCHIVO_CLUSTERS.exists() else None
    clusters, perfiles, metricas = cargar_datos(version)
    if clusters is not None:
        cfg.actualizar_calendario(ingesta.detectar_meses(clusters))

    if clusters is None:
        st.title("Planeación Cosnova")
        st.info("Los datos aún no están preparados. Solicita la actualización "
                "al responsable de la información.")
        return

    with st.sidebar:
        st.title("Planeación Cosnova")
        pagina = st.radio(
            "Navegacion",
            ["Inicio", "Analisis de Ventas", "Clusters", "Recomendacion de Compra",
             "Lanzamientos"],
            label_visibility="collapsed",
        )
        st.divider()

    if pagina == "Inicio":
        pagina_inicio(clusters, perfiles, metricas)
    elif pagina == "Analisis de Ventas":
        pagina_ventas(clusters)
    elif pagina == "Clusters":
        pagina_clusters(clusters, perfiles)
    elif pagina == "Recomendacion de Compra":
        pagina_recomendacion(clusters)
    else:
        pagina_lanzamiento(clusters)

    with st.sidebar:
        st.caption(
            f"Registros: {len(clusters):,}\n\n"
            f"Historico: {cfg.MESES[0]} a {cfg.MESES[-1]} ({cfg.TOTAL_MESES} meses)"
        )
        actualizado = datetime.fromtimestamp(version / 1_000_000_000,
                                             timezone(timedelta(hours=-5)))
        st.caption(f"Resultados actualizados el {actualizado:%d/%m/%Y %H:%M}.")


if __name__ == "__main__":
    main()
