# Clustering de Productos Cosnova — ESSENCE y CATRICE

Sistema para **planear la cantidad a pedir en cada compra semestral, por canal de
distribución**.

Son **dos modelos**, porque son dos problemas distintos:

| | Modelo de clustering | Modelo de lanzamientos |
|---|---|---|
| **Predice desde** | la historia de venta | los atributos del producto |
| **Sirve para** | productos que ya venden | productos que aún no tienen historia |
| **Técnica** | KMeans + factor semestral por cluster | regresión cuantílica + cohorte de análogos |
| **Entrega** | proyección S2 2026 | unidades para los primeros 6 meses de vida |
| **Cubre** | 1.430 registros | 385 registros |

El **clustering** agrupa los productos por cómo se venden (volumen, continuidad,
tendencia, estacionalidad, variabilidad) y usa el patrón semestral de cada grupo para
proyectar la siguiente compra.

El **modelo de lanzamientos** responde: *"voy a lanzar un labial tono Rojo, PVP
$23.000, grupo Labial, tipo Barra — ¿cuántas unidades pido?"*. Un producto a lanzar
tiene todas sus variables de comportamiento en cero, así que el clustering no puede
decir nada de él; hay que predecir desde lo único que se conoce antes de vender: qué
es el producto.

---

## 1. Resultados del modelo

Ejecutado sobre el archivo fuente entregado (`Metodo Planeación Cosnova News 2026 editado.xlsx`):

| Indicador | Valor |
|---|---|
| Registros producto × canal | 1.815 |
| Productos únicos (Material) | 605 |
| Canales | Exito, Farmatodo, Prosalon |
| Historia | 30 meses (ene-2024 a jun-2026) |
| Unidades vendidas en el histórico | 3.186.170 |
| Modelo seleccionado | **KMeans, k = 6** |
| Clusters de comportamiento | 6 + 1 grupo reservado (sin historial) |
| Semestre proyectado | **S2 2026** |
| Proyección S2 2026 (productos con historia) | 1.126.875 unidades |
| Sugerido de lanzamiento (productos sin historia) | 195.257 unidades |

### Los 6 clusters encontrados

| # | Nombre generado | Registros | Prom. mensual | Cobertura | Tendencia |
|---|---|---|---|---|---|
| 0 | Alta Rotación · En Crecimiento · Labios · Estacional S2 | 102 | 492 und | 76% | +457 |
| 1 | Alta Rotación · En Crecimiento · Ojos · Ambos Semestres | 397 | 135 und | 92% | +20 |
| 2 | Rotación Media · En Crecimiento · Rostro · Estacional S2 | 275 | 110 und | 37% | +68 |
| 3 | Rotación Media · En Declive · Ojos · Estacional S1 | 74 | 25 und | 54% | −7 |
| 4 | Bajo Movimiento · En Crecimiento · Rostro · Estacional S2 | 430 | 14 und | 55% | +15 |
| 5 | Bajo Movimiento · Ingreso Reciente · Labios · Ambos Semestres | 152 | 24 und | 10% | −0 |
| −1 | Producto Nuevo Sin Historial | 385 | — | 0% | — |

Los nombres se **derivan de los datos**, no están escritos a mano: si cambian los
datos o el `k` óptimo, los nombres se recalculan solos.

### Prioridad de compra asignada

| Prioridad | Registros | % |
|---|---|---|
| Alta | 410 | 22,6% |
| Media | 419 | 23,1% |
| Baja | 601 | 33,1% |
| Sin Movimiento | 385 | 21,2% |

---

## 2. Instalación

### Requisitos
- Python 3.10 o superior
- Windows, macOS o Linux

### El entorno virtual ya está creado

Vive en **`C:\venvs\cosnova`**, fuera de la carpeta del proyecto.

> **Por qué no está dentro del proyecto.** La ruta del proyecto tiene **175
> caracteres**. Un `.venv` aquí generaría rutas internas de hasta **272**, por encima
> del límite de **260** de Windows (las rutas largas están deshabilitadas en este
> equipo). La instalación de paquetes fallaría — de hecho ya pasó: el primer intento de
> instalar `jupyter` en este proyecto murió exactamente por eso. En `C:\venvs\cosnova`
> el problema desaparece y `notebook` instala sin problema.

Activarlo en la terminal de VS Code (PowerShell):

```powershell
C:\venvs\cosnova\Scripts\Activate.ps1
```

En CMD:

```
C:\venvs\cosnova\Scripts\activate.bat
```

En Git Bash:

```bash
source /c/venvs/cosnova/Scripts/activate
```

Para salir del entorno:

```bash
deactivate
```

**En VS Code no hace falta activarlo a mano:** `.vscode/settings.json` ya apunta al
intérprete y activa el entorno al abrir cada terminal nueva. Si VS Code no lo toma,
`Ctrl+Shift+P` → `Python: Select Interpreter` → `C:\venvs\cosnova\Scripts\python.exe`.

### Recrearlo desde cero

Si hace falta rehacerlo:

```powershell
python -m venv C:\venvs\cosnova
```

```powershell
C:\venvs\cosnova\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
C:\venvs\cosnova\Scripts\python.exe -m pip install -r requirements.txt
```

Verificar:

```powershell
C:\venvs\cosnova\Scripts\python.exe -c "import pandas, sklearn, streamlit, plotly, pyarrow; print('Instalacion correcta')"
```

### Cómo saber qué intérprete estás usando

El pipeline y las dos apps informan con qué Python están corriendo, para que usar el
del sistema por error no pase inadvertido.

Al ejecutar el pipeline:

```
==============================================================================
ENTORNO DE EJECUCION
==============================================================================
[INFO] Interprete: C:\venvs\cosnova\Scripts\python.exe
[INFO] Python 3.10.0 | entorno virtual 'cosnova'
```

Y si no es el entorno correcto, avisa explícitamente:

```
[ALERTA] No estas usando el entorno virtual del proyecto. Funciona igual porque
         el sistema tiene las mismas librerias, pero las versiones pueden
         diverger con el tiempo.
```

En las apps: el **dashboard analítico** lo muestra siempre en la barra lateral (en
rojo si es el del sistema); la **app de negocio** solo avisa cuando está mal, porque
al usuario final el intérprete no le aporta nada mientras sea el correcto.

> **Por qué importa este aviso.** El Python del sistema también tiene todas las
> librerías, así que ejecutar sin el entorno **funciona igual** — y por eso mismo el
> error sería invisible, hasta que alguna versión diverja y algo se comporte distinto
> sin explicación aparente. El aviso convierte un fallo silencioso en uno visible.

### Sobre las dependencias

La lista de `requirements.txt` sale de **escanear los imports reales** de todos los
`.py` y `.ipynb` del proyecto, no de suposiciones. Dos resultados de ese escaneo:

- **`yellowbrick` se eliminó.** Estaba declarado porque figuraba en el stack pedido al
  inicio, pero no se usa en ningún archivo: las curvas de codo y silueta están
  implementadas directamente en `src/clustering.py`.
- **`pyarrow` se conserva aunque no aparezca en ningún `import`.** Es el motor que
  pandas usa por debajo en `to_parquet()` / `read_parquet()`; sin él el pipeline falla
  al guardar `maestro.parquet`. Un "limpiar lo que no se importa" lo habría borrado y
  roto el proyecto.

---

## 3. Cómo abrir la aplicación (usuario final)

**Doble clic en `Iniciar_App.bat`.** Nada más. Se abre en el navegador.

| Archivo | Qué abre | Para quién |
|---|---|---|
| **`Iniciar_App.bat`** | App de 3 pasos: subir Excel → ver → descargar | comprador / planeación |
| `Iniciar_Dashboard_Analitico.bat` | Dashboard completo: clusters, ventas, métricas | analítica |

Los lanzadores se encargan de todo solos:

- Verifican que Python esté instalado y avisan con instrucciones si falta.
- Instalan las librerías la primera vez, si hacen falta.
- Evitan el prompt de correo que Streamlit muestra en su primera ejecución (si no
  se maneja, la consola se queda esperando una respuesta y la app nunca abre).
- El dashboard analítico calcula el modelo completo la primera vez (~2 min).
- La app de negocio prepara los datos sola si es la primera vez: el usuario nunca
  ve un mensaje de "ejecute primero el pipeline".

Para cerrar la aplicación, cierra la ventana negra de consola.

> **Alcance actual: local.** La app corre en el PC donde se abre. Para que otras
> personas entren desde sus equipos hay que publicarla — ver sección 12.

### Actualizar el modelo con meses nuevos de venta

Cuando se agrega al Excel fuente la venta POS de meses nuevos, la app lo detecta sola y
avisa arriba:

> **Hay 2 mes(es) de venta nuevos en el Excel que el modelo todavía no ha tomado en
> cuenta:** jul-26, ago-26. Los resultados de abajo están calculados sin esos meses.

Con el botón **🔄 Actualizar el modelo** se rehace toda la cadena (ingesta, features,
clustering, interpretación, lanzamientos y el Excel entregable) y se limpia la caché.
Tarda entre 1 y 2 minutos. El Excel fuente debe estar **cerrado**.

**El calendario es dinámico.** El proyecto ya no tiene una lista fija de meses: `ingesta`
lee los encabezados del archivo y actualiza los meses, los semestres, cuáles están
completos y cuál es el semestre a planear. Eso importa más de lo que parece — antes,
`config.MESES` era una lista fija de 30 meses y **todos** los módulos filtraban con
`[m for m in cfg.MESES if m in df.columns]`: un mes agregado al Excel se descartaba en
silencio y el modelo seguía calculando sobre los datos viejos sin avisar.

Por el mismo motivo se quitaron los nombres de semestre incrustados en el código
(`proyeccion_S2_2026` pasó a `proyeccion_semestre`, con el semestre concreto en la
columna `semestre_objetivo`): un nombre fijo habría mentido en cuanto el objetivo
cambiara de semestre.

### La app de negocio en 3 pasos

1. **Paso 1** — Descarga la plantilla (ya viene llena con ejemplos), reemplaza por tus
   productos y guarda.
2. **Paso 2** — Arrastra el archivo a la app.
3. **Paso 3** — Ve las cantidades sugeridas y presiona **DESCARGAR RESULTADOS EN EXCEL**.

También tiene una consulta rápida de un solo producto, sin necesidad de armar archivo.

---

## 3-bis. Ejecución técnica (analítica)

### Opción A — Pipeline completo (recomendado)

Un solo comando corre los 5 pasos y deja todos los artefactos listos:

```bash
python run_pipeline.py
```

Toma unos 25 segundos e imprime el avance de cada paso. Para omitir las gráficas
(y el t-SNE, que es lo más lento):

```bash
python run_pipeline.py --sin-figuras
```

### Opción B — Módulo por módulo

```bash
python -m src.ingesta
```

```bash
python -m src.feature_engineering
```

```bash
python -m src.clustering
```

```bash
python -m src.evaluation
```

```bash
python -m src.lanzamientos
```

```bash
python -m src.eda
```

### Opción C — Notebooks

```bash
jupyter notebook
```

Y abrir en orden: `01_EDA` → `02_Feature_Engineering` → `03_Clustering` →
`04_Resultados` → `05_Lanzamientos`.

Los notebooks explican cada decisión del modelo; los módulos de `src/` son la
implementación que se ejecuta en producción. **Ambos comparten el mismo código**:
los notebooks llaman a las funciones de `src/`, no las reimplementan.

### Dashboard interactivo

```bash
streamlit run dashboard/app.py
```

Abre en `http://localhost:8501`. Requiere haber corrido el pipeline al menos una
vez (el dashboard lee los resultados, no reentrena).

---

## 4. Estructura del proyecto

```
Cosnova_Clustering/
├── Iniciar_App.bat                  ← DOBLE CLIC: app para el usuario final
├── Iniciar_Dashboard_Analitico.bat  ← DOBLE CLIC: dashboard analítico
├── app/
│   ├── Sugerido_Lanzamientos.py     app de 3 pasos (subir → ver → descargar)
│   └── _preparar_entorno.py         evita el prompt de correo de Streamlit
├── .streamlit/
│   └── config.toml                  tema, sin telemetría, límite de carga
├── data/
│   ├── raw/                     Metodo Planeación Cosnova News 2026 editado.xlsx
│   ├── processed/               maestro.parquet, features.parquet
│   └── output/                  clusters.parquet, métricas, perfiles
├── notebooks/
│   ├── 01_EDA.ipynb             exploración y hallazgos de calidad de datos
│   ├── 02_Feature_Engineering.ipynb   construcción y validación de variables
│   ├── 03_Clustering.ipynb      k óptimo, 3 algoritmos y comparación
│   ├── 04_Resultados.ipynb      interpretación, proyección y entregable
│   └── 05_Lanzamientos.ipynb    sugerido para productos sin historial
├── src/
│   ├── config.py                rutas, constantes y parámetros del modelo
│   ├── ingesta.py               lectura, limpieza y unificación de canales
│   ├── feature_engineering.py   construcción de variables
│   ├── clustering.py            escalado, PCA, k óptimo y entrenamiento
│   ├── evaluation.py            perfiles, nombres, prioridad y proyección
│   ├── lanzamientos.py          sugerido para productos nuevos (modelo aparte)
│   ├── eda.py                   figuras del análisis exploratorio
│   └── utils.py                 utilidades y exportación del Excel final
├── dashboard/
│   └── app.py                   dashboard analítico Streamlit (5 páginas)
├── outputs/
│   ├── figures/                 14 figuras en PNG
│   └── Cosnova_Clusters_Final.xlsx    entregable para el negocio
├── run_pipeline.py              ejecuta todo de punta a punta
├── requirements.txt
└── README.md
```

---

## 5. El entregable: `Cosnova_Clusters_Final.xlsx`

| Hoja | Filas | Contenido |
|---|---|---|
| `Productos_Clasificados` | 1.815 | dataset completo con `Cluster`, `Nombre_Cluster` y `Prioridad_Compra` |
| `Perfiles_Cluster` | 7 | resumen estadístico de cada cluster |
| `Ventas_por_Semestre` | 1.816 | tabla producto × semestre con totales |
| `Recomendacion_Compra` | 1.815 | cantidad sugerida para S2 2026 por producto y canal |
| `Sugerido_Lanzamientos` | 385 | cantidad sugerida para productos sin historial |

En `Recomendacion_Compra` la columna **`metodo_proyeccion`** indica de dónde salió
cada cifra, para que el comprador sepa cuánta confianza darle.

> **Ojo con los 385 sin historial.** Aparecen en las dos hojas: en
> `Recomendacion_Compra` con **0** y método `Sin historial - ver hoja
> Sugerido_Lanzamientos`, y en `Sugerido_Lanzamientos` con su cifra estimada. El 0 de
> la primera hoja **no significa "no pedir"** — significa que el histórico no da base
> y que la cantidad está calculada en la otra hoja.

---

## 6. Cómo funciona el modelo

### Paso 1 — Ingesta

Lee las 3 hojas de canal, las unifica y limpia. **Cada fila del modelo es un
producto en un canal**: el mismo SKU en Exito y en Farmatodo son dos registros
distintos, porque su rotación y su estacionalidad también lo son.

### Paso 2 — Variables de comportamiento

Seis variables construidas desde los 30 meses de venta:

| Variable | Qué mide | Decisión de cálculo |
|---|---|---|
| `total_unidades` | volumen acumulado | suma de los 30 meses |
| `promedio_mensual` | intensidad de venta | se divide por **meses activos**, no por 30, para no castigar lanzamientos recientes |
| `cobertura_meses` | continuidad | % de los 30 meses con venta |
| `tendencia` | crece o cae | promedio S2-2025 − promedio S1-2024, en unidades/mes |
| `estacionalidad` | en cuál compra pedir | ratio S1/S2 sobre 2024 y 2025, acotado a [0,33 , 3] |
| `variabilidad` | qué tan predecible es | coef. de variación sobre **meses activos** |

Más los atributos del producto (subcategoría, marca, canal, tipo de portafolio)
codificados con One-Hot.

`ingreso_estimado` (unidades × PVP) **no entra al modelo**: es casi una
transformación lineal de `total_unidades` y contaría el volumen dos veces. Se
conserva para el reporte, donde sí es la cifra que importa.

### Paso 3 — Clustering

1. Se separan los productos sin una sola venta en 30 meses (cluster reservado −1).
   Todas sus variables valen cero: formarían un punto idéntico repetido cientos de
   veces, inflando las métricas y desplazando los centroides de los grupos reales.
2. `log1p` sobre las variables de volumen, luego `StandardScaler`.
3. PCA conservando el 90% de la varianza (24 variables → 9 componentes).
4. Se evalúan k de 2 a 12 con Elbow, Silhouette y Davies-Bouldin.
5. Se entrenan **KMeans**, **Jerárquico Ward** y **DBSCAN** (con `eps` calibrado
   automáticamente hasta que el ruido baje del 10%).
6. Gana el de mayor Silhouette **entre los que dejan menos del 10% sin clasificar**.

### Paso 4 — Proyección de S2 2026

Dos piezas.

**El factor de paso S1 → S2, por cluster.** Cuántas unidades vendió el cluster en
S2-2025 por cada unidad de S1-2025. Se calcula a nivel de grupo y no de producto
porque un SKU individual puede tener un semestre raro por un quiebre de stock o una
promoción puntual; el grupo promedia ese ruido. **Este es el aporte del modelo.**

**La jerarquía de métodos**, se aplica el primero que corresponda:

| # | Método | Cuándo | Registros |
|---|---|---|---|
| 1 | `S1_2026 × factor cluster` | vendió en el semestre más reciente | 1.407 (77,5%) |
| 2 | `Promedio S2 propio` | sin S1-2026, con S2 anteriores | 19 (1,1%) |
| 3 | `Promedio S2 del cluster` | con venta, sin ningún S2 propio | 4 (0,2%) |
| 4 | `Sin historial` | nunca vendió → lo define el área comercial | 385 (21,2%) |

El resultado se **redondea hacia arriba**: en una compra semestral un excedente
marginal cuesta menos que un quiebre de stock, que se lleva la venta completa.

### Prioridad de compra

| Prioridad | Condición | Acción |
|---|---|---|
| **Alta** | cobertura > 70% **y** tendencia positiva | asegurar abastecimiento |
| **Media** | cobertura 40–70% **o** tendencia estable | compra normal |
| **Baja** | cobertura < 40% **o** caída fuerte | revisar antes de recomprar |
| **Sin Movimiento** | nunca vendió | decisión comercial |

El orden de evaluación importa: un producto siempre disponible pero en caída fuerte
**no** sale como Alta solo por su cobertura, porque la condición de Alta exige
tendencia positiva.

---

## 6-bis. Sugerido para productos nuevos (`src/lanzamientos.py`)

Responde: *"voy a lanzar estos productos — ¿cuántas unidades pido de cada uno?"*

### A. Cargar un Excel con varios productos (uso principal)

En el dashboard, página **Producto Nuevo** → pestaña **Cargar Excel**. O por consola:

```bash
python -m src.lanzamientos --archivo mis_productos.xlsx
```

**Formato del archivo.** Una fila por producto. Columnas obligatorias:
`Subcategoria`, `Grupo`, `Tipo`, `PVP`. Opcionales: `Material`, `Descripcion`,
`Marca`, `Tono`, `mes_lanzamiento`, `Canal`.

Genera la plantilla llena de ejemplos con:

```bash
python -m src.lanzamientos --plantilla
```

**Deja `Canal` vacío** y recibes un sugerido para cada uno de los 3 canales — que es
como se hace la compra, y el canal explica el 47% de la predicción. Si pones un canal
específico, solo se calcula ese. Un archivo mixto (algunas filas con canal, otras sin)
también funciona.

**Tolerancia de formato.** Los encabezados se detectan solos: sirve `Subcategoría`,
`subcategoria` o `Linea`; `PVP`, `Precio` o `Precio de venta`. Los valores tampoco
necesitan estar perfectos: `liquido` → `Líquido`, `ROJO` → `Rojo`,
`Pestañinas` → `Pestañina`. Si falta el PVP se usa la mediana de su subcategoría; si
el mes es inválido se asume julio. **Todo lo que el sistema asume o corrige queda
listado en la hoja `Validacion`** del resultado, no se hace en silencio.

La detección de encabezados usa coincidencia **exacta** sobre el nombre normalizado,
no parcial. Con búsqueda parcial, `Tipo Portafolio` capturaría el alias `tipo` y el
modelo recibiría `New's Junio 2026` como si fuera la textura del producto — un error
silencioso que arruinaría todas las predicciones.

**Resultado:** un Excel de 4 hojas.

| Hoja | Contenido |
|---|---|
| `Sugerido_por_Canal` | una fila por producto × canal: P25/P50/P75 **a 6 y a 12 meses**, sugerido de ambos horizontes, desagregado `mes_01` a `mes_12`, confianza y comparables usados |
| `Resumen_por_Producto` | una fila por producto: total semestral y anual, con una columna por cadena de cada horizonte (`sem_Exito`, `anio_Exito`, …) y el desagregado mensual |
| `Plan_Mensual_Rotacion` | solo el plan mes a mes, limpio de columnas de diagnóstico, con totales de 6 y 12 meses y fila de TOTAL |
| `Validacion` | todo lo que el sistema asumió, corrigió o no reconoció |

Ejemplo de salida real:

```
                   Descripcion     Canal   PVP    P25    P50  sugerido      confianza
LABIAL ESSENCE ROJO INTENSO 3G     Exito 23000   49.7  191.3       250 Baja - revisar
LABIAL ESSENCE ROJO INTENSO 3G Farmatodo 23000  242.5  708.6       939           Alta
LABIAL ESSENCE ROJO INTENSO 3G  Prosalon 23000   45.1  142.1       221 Baja - revisar
PESTAÑINA CATRICE VOLUMEN 12ML Farmatodo 26000  512.0  790.0      1009          Media
```

La columna **`confianza`** es un semáforo derivado de qué tan específica pudo ser la
búsqueda de comparables: `Alta` (3+ atributos coincidieron), `Media` (2),
`Baja - revisar` (solo canal, o el modelo y los comparables discrepan más del doble).

### B. Consultar un solo producto

Dashboard → pestaña **Consultar un producto**, con formulario.

**Filtro en cascada:** al elegir la `Subcategoría`, el listado de `Grupo` se limita a
esa subcategoría (de 51 grupos a 7 en Labios, 4 en Uñas, 2 en Cejas).

`Tipo` y `Tono` **quedan con la lista completa a propósito**. Un producto nuevo puede
ser la primera combinación de su clase — por ejemplo el primer labial en barra de una
marca que hoy solo los tiene líquidos. Restringirlos bloquearía justamente los
lanzamientos más novedosos, que es lo que esta herramienta existe para estimar.

**El resultado se entrega en dos lecturas:**

| Lectura | Para qué |
|---|---|
| **Total 6 meses** (P25 / P50 / P75) | la cantidad de la primera compra semestral |
| **Por mes** | cómo escalonar la entrega y el abastecimiento |

Y desde código:

```python
from src import lanzamientos as lz
modelo = lz.entrenar()
lz.sugerir_lanzamiento(modelo, canal="Farmatodo", marca="Essence",
                       subcategoria="Labios", grupo="Labial", tipo="Barra",
                       tono="Rojo", pvp=23000, mes_lanzamiento=7)
```

### Los dos horizontes: 6 y 12 meses de rotación

El modelo entrega **dos cifras distintas**, cada una con su propio modelo entrenado:

| Horizonte | Para qué sirve | Set de entrenamiento |
|---|---|---|
| **6 meses** | la primera compra semestral | 916 lanzamientos |
| **12 meses** | el año completo de rotación | 686 lanzamientos |

**No se extrapola del semestre al año.** Se compararon las dos alternativas con
validación cruzada agrupada por Material:

| Método | MAE | MAE_log | Dentro de factor 2 |
|---|---|---|---|
| Mediana por canal (baseline) | 718 | 0,932 | 46,5% |
| Extrapolar 6m × ratio del canal | 546 | 0,659 | 62,8% |
| **Modelo directo a 12 meses** | **543** | **0,649** | **63,1%** |

Quedaron prácticamente empatados, y se eligió el directo por ser algo mejor y por no
encadenar dos estimaciones (el error de la primera no se arrastra a la segunda).
Cuantiles anuales calibrados: P25 cubre 28,6%, P50 49,4%, P75 70,0%.

**Por qué importaba no extrapolar con un factor único:** el ratio año/semestre no es 2,0
sino **2,13 de mediana**, y varía mucho por cadena — Éxito 1,93 · Farmatodo 1,94 ·
**Prosalón 2,75**. Un factor único habría subestimado Prosalón en un 40%.

**Coherencia garantizada:** como son dos modelos entrenados por separado, en
combinaciones raras el anual podría quedar por debajo del semestral. El código aplica un
piso (`anual >= semestral`) y reporta cuántas filas ajustó.

### El sugerido mes a mes

**La venta de un lanzamiento es bastante pareja**, con una leve subida hacia el final.
Reparto medido sobre los 916 lanzamientos históricos, ya con el primer mes corregido:

| Mes de vida | % del semestre |
|---|---|
| Mes 1 | 17,5% |
| Mes 2 | 15,4% |
| Mes 3 | 15,9% |
| Mes 4 | 17,5% |
| Mes 5 | 15,9% |
| Mes 6 | 17,9% |

Si fuera exactamente plano cada mes sería 16,7%, así que las diferencias son moderadas.
La app muestra las dos cifras —el promedio plano y el reparto con la curva real— para
que se vea de dónde sale cada una.

Los factores **se recalculan de los datos** en cada entrenamiento
(`calcular_reparto_mensual`), no están fijos en el código. El reparto usa el método de
los restos mayores para que **la suma cuadre exactamente con el total**: redondear cada
mes por separado inflaba la cifra (un sugerido de 1 unidad se convertía en 6).

**El reparto anual**, medido sobre los 686 lanzamientos con un año observable y ya
corregido el sesgo del primer mes:

| Mes | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % del año | 8,0 | 8,0 | 7,8 | 8,4 | 7,6 | 8,3 | 8,2 | 8,0 | 8,9 | 8,5 | 9,9 | 8,4 |

**Los meses 7 a 12 concentran el 51,9% del año**, contra 48,1% de los primeros seis. El
segundo semestre de vida de un lanzamiento vende algo *más* que el primero, con el pico
en el mes 11. Solo el 0,6% de los lanzamientos históricos muere en su segundo semestre.

### La corrección del primer mes

La curva cruda mostraba el mes 1 en **4,7%** y el mes 2 en 8,3%: casi el doble de un mes
al siguiente. Eso no ocurre en la realidad comercial, y efectivamente **era un artefacto
de medición, no demanda**.

**El origen.** El "mes 1" se define como el primer mes con venta, pero un producto que
entra al mercado el día 20 solo alcanza a vender diez días: ese mes queda medido a
medias y hunde el promedio de todos.

**La evidencia.** Separando los 686 lanzamientos según qué tan completo se ve su primer
mes:

| Grupo | n | Mes 1 vs nivel estable |
|---|---|---|
| Entraron tarde (mes1/mes2 < 0,5) | 227 | 0,17x |
| Intermedios | 203 | 0,67x |
| **Entraron temprano (mes1/mes2 ≥ 0,8)** | **233** | **1,00x** |

Los que sí tuvieron un mes completo venden **exactamente igual** que en los meses 2 a 6.
Además la serie salta 2,67x del mes 1 al 2 y luego se queda plana (ratios entre 0,79 y
1,31): eso es una discontinuidad puntual, no una rampa de demanda. Se descartaron las
otras explicaciones: un sell-in inicial habría hecho el mes 1 más *alto*, y el efecto no
depende del mes calendario de lanzamiento, así que tampoco es estacional.

**La corrección** (`_corregir_primer_mes`) estima el nivel de un primer mes completo
usando solo los lanzamientos que entraron temprano, y reemplaza el factor del mes 1. El
resto se reescala para que la curva siga sumando 1: **el total anual lo fija el modelo y
no se infla por este ajuste**. El factor se recalcula de los datos en cada entrenamiento.

Efecto en un sugerido de 1.775 unidades:

```
              Mes 1   Mes 2   Mes 3   Mes 4
Antes            83     147     144     155     <- el mes 2 casi duplicaba al 1
Después         142     141     139     150     <- estable, como en la realidad
```

> **Un matiz honesto:** si el producto se lanza justo al inicio de un mes, sus 12 meses
> de rotación son 12 meses completos y el total real podría quedar un ~3% por encima del
> que estima el modelo, porque los lanzamientos con que se entrenó incluían ese primer
> mes parcial. Se prefirió no inflar el total y solo repartirlo bien.

### C. Los pendientes del archivo fuente

```bash
python -m src.lanzamientos
```

Entrena, valida y puntúa los 385 registros sin historial que ya vienen en el Excel
fuente. También corre dentro de `run_pipeline.py`.

### Resultado del ejemplo

```
ESTIMACION PARA LOS PRIMEROS 6 MESES
  Escenario bajo   (P25):      242 unidades
  Escenario medio  (P50):      709 unidades
  Escenario alto   (P75):      938 unidades

>>> SUGERIDO DE COMPRA: 939 unidades  (P75)
    Ingreso proyectado: $ 21.597.000

RESPALDO: 10 productos comparables
  Criterio de busqueda: Canal + Grupo + Tipo + PVP +-25%
  Su venta real: P25=123 | P50=420 | P75=752
  Modelo y comparables coinciden (diferencia 1.25x)
```

### Qué predice

`und_primeros_6m`: unidades vendidas en los **primeros 6 meses de vida** del producto
en ese canal — exactamente lo que se pide en la primera compra semestral.

### El set de entrenamiento: 916 de 1.815 registros

El filtro es la decisión más delicada del modelo:

| Grupo | Registros | Destino |
|---|---|---|
| Nunca vendieron | 385 | son los que hay que **predecir** |
| Ya vendían en ene-24 | 352 | **excluidos** (censura a la izquierda) |
| Debut hace menos de 6 meses | 162 | excluidos (target incompleto) |
| **Debut limpio + 6 meses observables** | **916** | **entrenamiento** |

**Por qué se excluyen los 352 censurados.** Enero 2024 es el primer mes de la ventana
de datos. Un producto que ya vendía ahí pudo haber lanzado en 2023, 2020 o 2015: su
debut real es desconocido. Tomar sus primeros 6 meses observados como si fuera un
lanzamiento mediría un producto en régimen, no un arranque. Serían **352 etiquetas
equivocadas sobre 1.268** — suficiente para inflar sistemáticamente todos los sugeridos.

### Dos motores que se cruzan

**1. Modelo cuantílico** (Gradient Boosting con `loss="quantile"`): entrega P25, P50 y
P75 directamente. Es la cifra oficial.

**2. Cohorte de análogos**: busca productos históricos parecidos en cascada, de lo más
específico a lo más general, y se detiene al reunir 5 comparables:

| # | Criterio |
|---|---|
| 1 | `Canal + Marca + Grupo + Tipo + Tono + PVP ±20%` |
| 2 | `Canal + Marca + Grupo + Tipo + Tono` |
| 3 | `Canal + Marca + Grupo + Tono` — suelta Tipo, conserva Marca y Tono |
| 4 | `Canal + Grupo + Tipo + Tono` — suelta Marca, conserva Tono |
| 5 | `Canal + Grupo + Tono` |
| 6 | `Canal + Marca + Grupo + Tipo` — ya sin Tono, reintenta con Marca |
| 7 | `Canal + Marca + Grupo` |
| 8 | `Canal + Grupo + Tipo` |
| 9 | `Canal + Grupo` |
| 10 | `Canal + Subcategoría + Tono` |
| 11 | `Canal + Subcategoría + Tipo` |
| 12 | `Canal + Subcategoría` |
| 13 | `Canal` |

**Por qué ese orden de prioridad.** No es arbitrario, sale de medir cuánto cambia la
venta al variar cada atributo sobre los 916 lanzamientos históricos:

- **`Canal` nunca se suelta** — explica el 47% de la predicción (mediana de debut: 709
  und en Farmatodo vs 69 en Prosalon).
- **`Tono` se suelta tarde** — controlando por grupo, la mediana varía **2–3x**
  (Brillo 3,0x · Pestañina 2,8x · Rubor 2,3x · Labial 2,1x).
  *Ojo con la dirección:* dentro de Brillo el café vende **más** que el rosado (606 vs
  267); globalmente es al revés (176 vs 239) porque el rosado abunda en Rubor. El tono
  pesa mucho, pero no en un sentido fijo.
- **`Marca` se suelta antes que Tono**, aunque se incluye: comparar un lanzamiento
  Essence contra productos Catrice desorienta al comprador.
- **`Tipo` se suelta primero** entre los atributos de producto: resultó el menos
  discriminante.

**El criterio alcanzado es la medida de confianza.** El resultado reporta además dos
banderas explícitas: `marca_respetada` y `tono_respetado`, con el porcentaje de
comparables que efectivamente comparten cada atributo. Los comparables se muestran
ordenados por parecido — misma marca primero, luego mismo tono.

**El mínimo de comparables es 5, no 8.** Se bajó al agregar Marca y Tono a la cascada:
exigiendo 8, el nivel más específico solo se alcanzaba en el 9,7% de los casos, así que
Marca y Tono se soltaban casi siempre y el cambio no habría servido de nada. El costo es
real: un P75 sobre 5 puntos es ruidoso. Por eso **el número oficial lo da el modelo
cuantílico** (que usa las 916 filas) y la cohorte funciona como respaldo visible y
contraste; `cohorte_n` siempre reporta el tamaño.

Cobertura lograda sobre los 385 pendientes: **marca respetada en 47,8%** (antes 0%, no
estaba en la cascada), **tono respetado en 37,4%**, ambos en 10,1%. Ese techo lo impone
el tamaño del set de entrenamiento, no la lógica.

Si los dos motores difieren por más del doble, la fila se marca en
`requiere_revision`: hay que validarla comercialmente antes de pedir. Hoy son **130 de
385 registros (34%)**.

### Validación

GroupKFold por `Material`, 5 folds:

| Método | MAE | MAE_log | Dentro de factor 2 |
|---|---|---|---|
| Mediana por canal (baseline) | 312 | 0,916 | 44,2% |
| **Modelo de atributos** | **227** | **0,628** | **63,9%** |

**+31% de mejora** sobre lo que haría un analista sin modelo.

**Por qué GroupKFold y no KFold normal.** El mismo `Material` aparece en los 3 canales
con targets correlacionados. Con KFold simple, Exito puede caer en train y Farmatodo en
test: el modelo "acierta" por fuga de información y **la métrica mentiría**. Agrupar
por Material cierra esa fuga.

**Calibración de los cuantiles** (que el P75 cubra de verdad ~75%):

| Cuantil | Cobertura objetivo | Cobertura real |
|---|---|---|
| P25 | 25% | 29,3% |
| P50 | 50% | 49,0% |
| P75 | 75% | 71,8% |

### Qué atributos pesan

| Variable | Importancia |
|---|---|
| Canal Farmatodo | 0,468 |
| PVP relativo a su subcategoría | 0,074 |
| Mes de lanzamiento | 0,064 |
| Marca | 0,106 (Essence + Catrice) |
| Subcategoría, Grupo, Tipo, Tono | el resto |

**El canal domina con el 47%.** Un lanzamiento en Farmatodo vende ~10x lo que el mismo
producto en Prosalon (mediana de debut: 709 vs 69 unidades). Implicación práctica: la
conversación de planeación debería empezar por *"cuánto en cada canal"* y solo después
por *"qué producto"*.

El **PVP relativo** resultó más informativo que el PVP crudo (0,074 vs 0,033): un labial
de $23.000 no es caro en absoluto, es caro o barato *frente a los otros labiales*.

### Límites de este modelo

1. **Acierta dentro del factor 2 en el 64% de los casos.** Es una estimación con
   incertidumbre real, no una cifra exacta. Por eso se entrega el rango P25–P50–P75.
2. **No conoce el apoyo comercial**: plan de medios, exhibición negociada, respaldo
   publicitario. Dos productos idénticos con distinto apoyo venden muy distinto y el
   modelo no puede verlo. **Es su límite más importante.**
3. **Un tono o grupo nunca visto** se predice apoyándose en el resto de los atributos,
   con menos precisión. La señal de revisión avisa cuando eso pasa.
4. Con más años de historia, los 352 censurados se recuperarían y el modelo mejoraría.

---

## 7. Hallazgos sobre la fuente

Cinco cosas que el archivo real trae y que el pipeline maneja explícitamente:

1. **El Excel tiene 6 hojas, no 3.** `BD`, `Ventas` (980 mil filas) y `TD POS` son
   insumos previos del negocio y no entran al modelo. Solo se leen las 3 hojas de canal.

2. **Los encabezados de mes son fechas, no texto.** Vienen como `2024-01-01`, no
   como `ene-24`. Si se asumiera texto, el pipeline no encontraría ni una columna de venta.

3. **`Status` viene escrito de tres formas** para el mismo concepto: `Ex artcile`
   (con typo), `ex article` y `Ex article`. Filtrar por texto exacto dejaría dentro
   111 registros descontinuados. La comparación se hace sobre texto normalizado.

4. **`EAN/UPC-Code` tiene un duplicado real**: el código `4059729583222` aparece en
   un material Catrice y en uno Essence. No es un error deduplicable — son dos
   productos distintos. Por eso la clave del modelo es **`Material` + `Canal`**.

5. **`Categoría` es constante** (`Maquillaje` en las 1.815 filas). No aporta nada al
   clustering y el pipeline la descarta automáticamente. La jerarquía informativa
   real es `Subcategoría` → `Grupo` → `Tono`.

6. **`Grupo` trae el mismo valor en singular y plural** como si fueran categorías
   distintas. Los pares verificados y su peso:

   | Par | Registros |
   |---|---|
   | `Lápiz de ojos` / `Lápices de ojos` | 108 / 21 |
   | `Pestañina` / `Pestañinas` | 126 / 30 |
   | `Corrector` / `Correctores` | 75 / 15 |
   | `Base` / `Bases` | 63 / 15 |
   | `Polvo` / `Polvos` | 48 / 18 |
   | `Lápiz de cejas` / `Lápices de cejas` | 12 / 21 |
   | `Brocha` / `Brochas` | 3 / 3 |

   Esto fragmentaba las cohortes de productos comparables sin ninguna razón de
   negocio. El pipeline los unifica (`config.SINONIMOS_GRUPO`), pasando de 58 a 51
   niveles. El efecto medido: **212 filas hacen match por su grupo real en lugar de
   164** (+29%), y las que caían a `Subcategoría` genérica bajaron de 198 a 150.

   No se unifican `Tinta`, `Tinta labial` y `Tinta de labios`: podrían ser tinta de
   labios o de uñas y el dato no permite saberlo.

7. **Tres valores de `Tipo` y 13 de `Grupo` no existen en el set de entrenamiento** de
   lanzamientos, porque todos sus productos nunca vendieron o están censurados. El más
   relevante es `Con aplicador`: 162 registros en el maestro, de los cuales 127 nunca
   vendieron. Un lanzamiento con ese tipo se predice apoyándose en los demás atributos
   y el sistema lo avisa explícitamente en la hoja `Validacion`.

Y dos hallazgos de negocio:

- **Farmatodo concentra el 78% del volumen** (Exito 12%, Prosalon 9%). El canal es
  un factor de primer orden en la planeación.
- **El segundo semestre vende más que el primero**, de forma consistente en 2024 y
  2025, con diciembre como pico. Por eso la proyección aplica un factor S1 → S2 en
  lugar de un promedio simple.

---

## 8. Limitaciones

Conviene tenerlas presentes al usar el resultado:

1. **El modelo solo ve historia de venta.** No conoce quiebres de stock,
   promociones, cambios de exhibición ni acciones de la competencia. Un producto que
   vendió poco por falta de inventario se ve igual que uno que vendió poco por falta
   de demanda. **Es la limitación más importante.**

2. **El Silhouette es moderado (≈ 0,20).** Los productos forman un continuo más que
   grupos nítidamente separados. Los clusters son útiles como segmentos de política
   de compra, no como categorías naturales indiscutibles. Las medias por cluster sí
   se diferencian claramente en las variables que importan.

3. **Los productos sin historial no se proyectan.** Son el 21% de los registros. En
   su mayoría no son fracasos comerciales sino novedades que aún no han entrado a
   vender en ese canal.

4. **El factor de cluster está acotado a [0,5x , 2,0x].** Los clusters cuyo factor
   real toca el techo quedan con una proyección conservadora a propósito, para evitar
   que un grupo de base pequeña dispare la recomendación.

5. **Solo hay 2 años completos** para estimar estacionalidad. Con un tercer año la
   estimación mejorará.

### Siguiente iteración

- Incorporar días sin stock, para distinguir demanda real de venta perdida.
- Marcar los meses con promoción, para no leer un pico promocional como tendencia.
- Volver a correr el pipeline al cierre de cada semestre: el factor de paso se
  recalibra automáticamente con el dato nuevo.

---

## 9. Figuras generadas

En `outputs/figures/`:

| Archivo | Contenido |
|---|---|
| `01_distribucion_por_dimension.png` | ventas por canal, marca, subcategoría y grupo |
| `02_evolucion_mensual.png` | serie de 30 meses, total y por canal |
| `03_heatmap_productos.png` | mapa de calor top 50 productos × mes |
| `04a_distribucion_cobertura.png` | histograma y tramos de cobertura |
| `04b_boxplots_volumen.png` | dispersión del volumen por subcategoría y canal |
| `04c_tendencias.png` | productos en crecimiento vs. en declive |
| `04d_correlaciones.png` | matriz de correlación (Spearman) |
| `05_scatter_pca_clusters.png` | clusters en el espacio PCA |
| `06_tsne_clusters.png` | proyección t-SNE |
| `07_perfil_clusters.png` | composición de cada cluster |
| `08_metricas_por_cluster.png` | métricas promedio por cluster |
| `09_dendrograma.png` | dendrograma del modelo jerárquico |
| `10_importancia_atributos_lanzamiento.png` | qué atributos explican el volumen de un lanzamiento |
| `11_distribucion_debut_lanzamientos.png` | distribución del volumen de debut, por canal |

---

## 10. Solución de problemas

**`ModuleNotFoundError: No module named 'src'`**
Ejecuta desde la **raíz del proyecto**, no desde una subcarpeta. Los notebooks
resuelven la ruta solos.

**`FileNotFoundError` con el archivo Excel**
El Excel debe estar en `data/raw/`. Verifica:

```bash
python -c "from src import config; print(config.ARCHIVO_EXCEL.exists())"
```

**El dashboard dice "Todavía no hay resultados del modelo"**
Falta correr el pipeline: `python run_pipeline.py`.

**`PermissionError: [Errno 13] Permission denied` al leer el Excel fuente**
El archivo está abierto en Excel, que lo bloquea. Ciérralo y vuelve a ejecutar.

**Mi Excel de lanzamientos da error de columnas obligatorias**
El archivo necesita al menos `Subcategoria`, `Grupo`, `Tipo` y `PVP`. Descarga la
plantilla con `python -m src.lanzamientos --plantilla` y compara los encabezados. Ojo:
`Tipo Portafolio` **no** sirve como `Tipo` — son cosas distintas (uno es la campaña, el
otro la textura del producto).

**Muchas filas salen con confianza "Baja - revisar"**
No es un error: significa que esa combinación de atributos tiene pocos comparables
históricos. Revisa la columna `cohorte_nivel` para ver qué tan específica pudo ser la
búsqueda, y prioriza validar comercialmente las que se quedaron en `Canal` o
`Canal + Subcategoría`.

**Errores de instalación por rutas demasiado largas (`OSError: [Errno 2] No such file or directory`)**
Es el límite de 260 caracteres de Windows. Por eso el entorno virtual vive en
`C:\venvs\cosnova` y no dentro del proyecto. Si aun así aparece, se pueden habilitar
las rutas largas (requiere permisos de administrador):

```
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**VS Code no reconoce las librerías (subrayado rojo en los imports)**
No está usando el intérprete del entorno. `Ctrl+Shift+P` → `Python: Select Interpreter`
→ `C:\venvs\cosnova\Scripts\python.exe`. Luego recarga la ventana con
`Ctrl+Shift+P` → `Developer: Reload Window`.

**`ModuleNotFoundError` al ejecutar desde la terminal**
Estás usando el Python del sistema en vez del entorno. Actívalo primero
(`C:\venvs\cosnova\Scripts\Activate.ps1`) o llama al intérprete completo:

```powershell
C:\venvs\cosnova\Scripts\python.exe run_pipeline.py
```

**Los resultados cambian entre ejecuciones**
No deberían: la semilla está fijada en `config.SEMILLA = 42` para KMeans, PCA y
t-SNE. Si cambian, es porque cambió el archivo fuente.

---

## 11-bis. Publicar la app para más usuarios

Hoy la app es **local**: corre en el PC donde se abre el `.bat`. Si más personas la
necesitan, estas son las opciones en orden de esfuerzo:

**A. Compartir en la red interna (sin instalar nada nuevo).** Se arranca con la IP
visible y el equipo entra desde su navegador:

```bash
python -m streamlit run app/Sugerido_Lanzamientos.py --server.address 0.0.0.0 --server.port 8501
```

Los demás entran a `http://<tu-ip>:8501` (la IP aparece como *Network URL* al
arrancar). Requiere estar en la misma red, permitir el puerto en el Firewall de
Windows, y que tu PC esté encendido. Los datos nunca salen de la red de Prebel.

**B. Servidor interno.** Igual que A pero en un servidor de la empresa, así no depende
de un PC personal. Requiere apoyo de TI.

**C. Nube pública (Streamlit Community Cloud, gratis).** Da una URL accesible desde
cualquier parte. **Implica subir el histórico de ventas de Cosnova a un servidor de un
tercero**, así que antes hay que tener aprobación de TI/seguridad y del propio cliente.
No lo hagas sin ese permiso.

**D. Ejecutable `.exe`.** Se empaqueta con PyInstaller para instalar en otros PCs sin
Python. Es la opción más parecida a un programa tradicional, pero el archivo pesa
cientos de MB y cada actualización del modelo obliga a redistribuirlo.

---

## 12. Configuración

Todos los parámetros del modelo están en `src/config.py`. Los más útiles de ajustar:

| Parámetro | Valor | Qué controla |
|---|---|---|
| `K_MIN` / `K_MAX` | 2 / 12 | rango de búsqueda del número de clusters |
| `VARIANZA_PCA` | 0.90 | varianza que conserva el PCA |
| `MAX_RUIDO_DBSCAN` | 0.10 | techo de ruido tolerado |
| `UMBRAL_COBERTURA_ALTA` | 70.0 | corte de cobertura para prioridad Alta |
| `UMBRAL_COBERTURA_BAJA` | 40.0 | corte de cobertura para prioridad Baja |
| `SEMESTRE_OBJETIVO` | `S2_2026` | semestre que se proyecta |
| `SEMILLA` | 42 | reproducibilidad |

Al cerrar un semestre nuevo: agregar los meses a `MESES` y `SEMESTRES`, actualizar
`SEMESTRE_OBJETIVO` y `SEMESTRES_HOMOLOGOS`, y volver a correr el pipeline.
