"""Descarga ventas POS por mes y NIT usando el adaptador SAP de Elasticidades."""

import argparse
import calendar
import importlib.util
import json
import os
import io
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent


def mes(texto):
    try:
        return date.fromisoformat(texto + "-01")
    except ValueError as error:
        raise argparse.ArgumentTypeError("Usa el formato AAAA-MM.") from error


def guardar_atomico(df, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destino.parent, suffix=".parquet", delete=False) as tmp:
        provisional = Path(tmp.name)
    try:
        df.to_parquet(provisional, index=False)
        os.replace(provisional, destino)
    finally:
        provisional.unlink(missing_ok=True)


def agregar_ean(df, servicio):
    """Cruza el maestro CDS de Elasticidades sin multiplicar las ventas."""
    import pandas as pd

    materiales = df["Material-CHAVL"].astype(str).str.strip()
    with redirect_stdout(io.StringIO()):
        maestro, errores = servicio.query_cds(
            materiales.unique().tolist(), "MATERIAL,EAN", "ZMMS020MATERIAL", "MATERIAL"
        )
    if errores:
        print(f"CDS reportó errores para {len(errores):,} materiales. Revisa los EAN faltantes.")
    if maestro.empty:
        maestro = pd.DataFrame(columns=["MATERIAL", "EAN"])
    maestro = maestro[["MATERIAL", "EAN"]].copy()
    maestro["MATERIAL"] = maestro["MATERIAL"].astype("string").str.strip()
    maestro["EAN"] = maestro["EAN"].astype("string").str.strip()
    maestro = maestro[maestro["EAN"].notna() & maestro["EAN"].ne("")]
    maestro = maestro.drop_duplicates(subset="MATERIAL", keep="first")
    resultado = df.drop(columns="ean", errors="ignore").copy()
    resultado["ean"] = materiales.map(maestro.set_index("MATERIAL")["EAN"]).fillna("Sin EAN")
    print(f"Registros sin EAN: {resultado['ean'].eq('Sin EAN').sum():,} de {len(resultado):,}.")
    return resultado


def extraer_bloque(conexion, config, params):
    """Reconstruye las celdas por ordinal y comprueba las medidas de la vista."""
    import pandas as pd

    requeridas = ["Día natural-CHAVL", "Material-CHAVL", "Nit-CHAVL",
                  *config["column_names"]]
    try:
        raw = conexion.execute_query(config["query_name"], config["view_id"], params)
    except Exception as error:
        # BW expresa también la ausencia de ventas como una excepción RFC.
        if getattr(error, "key", None) == "NO_APPLICABLE_DATA":
            return pd.DataFrame(columns=requeridas)
        raise
    ejes = {eje["AXIS"]: eje["SET"] for eje in raw["E_AXIS_DATA"]}
    if not ejes.get("001"):
        return pd.DataFrame(columns=requeridas)
    medidas = sorted(ejes["000"], key=lambda item: int(item["TUPLE_ORDINAL"]))
    nombres = [item["CAPTION"] for item in medidas]
    if nombres != config["column_names"]:
        raise ValueError(f"Las medidas SAP cambiaron. Se recibieron {nombres}.")
    info = conexion.extract_axis_info(raw["E_AXIS_INFO"])
    dimensiones = conexion.clean_data(raw["E_AXIS_DATA"], raw["E_CELL_DATA"])
    filas = conexion.data_structuring(dimensiones, info, ["CAPTION", "CHAVL"])
    filas.index = filas["TUPLE_ORDINAL"].astype(int)
    celdas = pd.DataFrame(raw["E_CELL_DATA"])
    if celdas.empty:
        raise ValueError("SAP devolvió dimensiones sin celdas de ventas.")
    ordinal = pd.to_numeric(celdas["CELL_ORDINAL"], errors="raise").astype(int)
    if ordinal.duplicated().any() or (ordinal < 0).any():
        raise ValueError("SAP devolvió ordinales de celdas inválidos.")
    celdas["fila"] = ordinal // len(nombres)
    celdas["medida"] = ordinal % len(nombres)
    if not celdas["fila"].isin(filas.index).all():
        raise ValueError("Hay celdas SAP sin una fila correspondiente.")
    valores = celdas.pivot(index="fila", columns="medida", values="VALUE")
    valores = valores.reindex(index=filas.index, columns=range(len(nombres)))
    if valores.isna().any().any():
        raise ValueError("SAP devolvió celdas incompletas.")
    valores.columns = nombres
    return pd.concat([filas, valores], axis=1).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", type=mes, required=True)
    parser.add_argument("--fin", type=mes, required=True)
    parser.add_argument("--meses-por-bloque", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--credenciales", type=Path, required=True,
                        help="JSON con ashost, sysnr, client, user, passwd y lang.")
    parser.add_argument("--config", type=Path, default=ROOT / "sap_zpos_config.json")
    parser.add_argument("--nit", action="append", help="Repetible. Por defecto usa los NIT del M.")
    parser.add_argument("--salida", type=Path, default=ROOT / "data/raw/ventas_pos_sap.parquet")
    args = parser.parse_args()
    if args.inicio > args.fin:
        parser.error("--inicio debe ser anterior o igual a --fin.")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    nits = list(dict.fromkeys(args.nit or config["nits"]))
    if not nits or any(not isinstance(nit, str) or not nit.isdigit() for nit in nits):
        parser.error("Los NIT deben ser textos con solo dígitos.")
    credenciales = json.loads(args.credenciales.read_text(encoding="utf-8-sig"))
    requeridas = {"ashost", "sysnr", "client", "user", "passwd", "lang"}
    if set(credenciales) != requeridas or any(
        not isinstance(v, str) or not v.strip() for v in credenciales.values()
    ):
        parser.error("El JSON debe contener ashost, sysnr, client, user, passwd y lang como textos.")

    # Carga el mismo adaptador utilizado por el proyecto de Elasticidades.
    spec = importlib.util.spec_from_file_location("cosnova_sap_adapter", config["adapter_path"])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    cds_path = config.get("cds_adapter_path", str(Path(config["adapter_path"]).with_name("cds_adapter.py")))
    cds_spec = importlib.util.spec_from_file_location("cosnova_cds_adapter", cds_path)
    cds_adapter = importlib.util.module_from_spec(cds_spec)
    cds_spec.loader.exec_module(cds_adapter)
    import pandas as pd

    conexion = adapter.SAPConnection(**credenciales)
    bloques = []
    actual = args.inicio
    try:
        while actual <= args.fin:
            indice_final = actual.year * 12 + actual.month - 1 + args.meses_por_bloque - 1
            anio_final, mes_final = divmod(indice_final, 12)
            mes_final += 1
            limite = min(date(anio_final, mes_final, 1), args.fin)
            ultimo = date(limite.year, limite.month, calendar.monthrange(limite.year, limite.month)[1])
            for nit in nits:
                print(f"Consultando {actual:%Y-%m} a {ultimo:%Y-%m}, NIT {nit}", flush=True)
                params = [
                    ("VAR_ID_6", "0I_DAYIN                      0004"),
                    ("VAR_VALUE_LOW_EXT_6", actual.strftime("%d.%m.%Y")),
                    ("VAR_VALUE_HIGH_EXT_6", ultimo.strftime("%d.%m.%Y")),
                    ("VAR_ID_9", "0TAX_NUMB_V010                0004"),
                    ("VAR_VALUE_EXT_9", nit),
                ]
                df = extraer_bloque(conexion, config, params)
                necesarias = {"Día natural-CHAVL", "Material-CHAVL", "Nit-CHAVL",
                              *config["column_names"]}
                if not isinstance(df, pd.DataFrame) or not necesarias.issubset(df.columns):
                    raise ValueError("La respuesta SAP carece de las columnas esperadas.")
                if df.empty:
                    print("Sin registros.", flush=True)
                    continue
                fechas = pd.to_datetime(df["Día natural-CHAVL"].astype(str), format="%Y%m%d")
                if not fechas.between(pd.Timestamp(actual), pd.Timestamp(ultimo)).all():
                    raise ValueError("SAP devolvió fechas fuera del rango solicitado.")
                if not df["Nit-CHAVL"].astype(str).str.strip().eq(nit).all():
                    raise ValueError("SAP devolvió NIT diferentes al solicitado.")
                for columna in config["column_names"]:
                    df[columna] = pd.to_numeric(df[columna], errors="raise")
                    if df[columna].isna().any():
                        raise ValueError(f"Hay valores vacíos en {columna}.")
                df["fecha"] = fechas
                df["Material-CHAVL"] = df["Material-CHAVL"].astype(str)
                bloques.append(df)
                print(f"{len(df):,} filas recibidas.", flush=True)
            actual = ultimo + timedelta(days=1)
    finally:
        conexion.close_connection()
    if not bloques:
        raise ValueError("La consulta completa no devolvió ventas. Revisa fechas, vista y NIT.")
    resultado = pd.concat(bloques, ignore_index=True)
    servicio = cds_adapter.CoreDataService(
        base_url=config["cds_url"], batch_mode=True, batch_size=150
    )
    try:
        resultado = agregar_ean(resultado, servicio)
    finally:
        servicio.session.close()
    guardar_atomico(resultado, args.salida)
    print(f"Guardado {args.salida.resolve()} con {len(resultado):,} filas.")


if __name__ == "__main__":
    main()
