import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.simulacion import aplicar_ajustes, construir_plan


class SimulacionMensualTests(unittest.TestCase):
    def setUp(self):
        self.modelo = SimpleNamespace(
            reparto_mensual=np.full(6, 1 / 6),
            reparto_mensual_anual=np.full(12, 1 / 12),
        )
        self.productos = pd.DataFrame({
            "Material": ["A", "A"], "Canal": ["Exito", "Farmatodo"],
            "mes_lanzamiento": [11, 12], "unidades_sugeridas": [60, 30],
            "unidades_sugeridas_anual": [120, 60],
        })

    def ajustes(self, plan):
        return pd.DataFrame({
            "Mes": sorted(plan["Mes"].unique()),
            "Peso estacional": 1.0, "Ajuste comercial (%)": 0.0, "Motivo": "",
        })

    def test_calendario_y_escenario_neutro_en_ambos_horizontes(self):
        for horizonte in [6, 12]:
            plan = construir_plan(self.productos, self.modelo, horizonte, 2026)
            ajustes = self.ajustes(plan)
            resultado = aplicar_ajustes(plan, ajustes)
            self.assertEqual(plan.iloc[0]["Mes"], "2026-11")
            self.assertEqual(plan.iloc[horizonte]["Mes"], "2026-12")
            self.assertIn("2027-01", plan["Mes"].values)
            pd.testing.assert_series_equal(
                resultado["Base del modelo"], resultado["Unidades ajustadas"],
                check_names=False,
            )
            self.assertNotIn("Unidades ajustadas", plan)

    def test_pesos_conservan_el_total_por_producto_y_canal(self):
        plan = construir_plan(self.productos, self.modelo, 6, 2026)
        ajustes = self.ajustes(plan)
        ajustes.loc[ajustes["Mes"] == "2026-12", "Peso estacional"] = 3
        resultado = aplicar_ajustes(plan, ajustes)
        totales = resultado.groupby("Registro")[
            ["Base del modelo", "Con estacionalidad", "Unidades ajustadas"]
        ].sum()
        self.assertEqual(totales["Base del modelo"].tolist(), [60, 30])
        self.assertEqual(totales["Con estacionalidad"].tolist(), [60, 30])
        self.assertEqual(totales["Unidades ajustadas"].tolist(), [60, 30])
        diciembre = resultado[resultado["Mes"] == "2026-12"]
        self.assertTrue((diciembre["Con estacionalidad"] > diciembre["Base del modelo"]).all())

    def test_aumento_y_reduccion_comercial(self):
        plan = construir_plan(self.productos.iloc[:1], self.modelo, 6, 2026)
        ajustes = self.ajustes(plan)
        ajustes.loc[ajustes["Mes"] == "2026-11", "Ajuste comercial (%)"] = 15
        ajustes.loc[ajustes["Mes"] == "2026-12", "Ajuste comercial (%)"] = -100
        resultado = aplicar_ajustes(plan, ajustes)
        self.assertEqual(resultado["Unidades ajustadas"].tolist(), [12, 0, 10, 10, 10, 10])
        self.assertEqual(resultado["Unidades ajustadas"].sum(), 52)

    def test_volumen_pequeno_y_cero(self):
        productos = self.productos.copy()
        productos["unidades_sugeridas"] = [1, 0]
        plan = construir_plan(productos, self.modelo, 6, 2026)
        ajustes = self.ajustes(plan)
        ajustes.loc[ajustes["Mes"] == "2027-03", "Peso estacional"] = 100
        resultado = aplicar_ajustes(plan, ajustes)
        self.assertEqual(resultado.groupby("Registro")["Unidades ajustadas"].sum().tolist(), [1, 0])
        self.assertEqual(resultado.loc[resultado["Mes"] == "2027-03", "Unidades ajustadas"].sum(), 1)

    def test_entradas_invalidas(self):
        plan = construir_plan(self.productos, self.modelo, 6, 2026)
        for columna, valor in [
            ("Peso estacional", 0), ("Peso estacional", -1),
            ("Peso estacional", np.nan), ("Ajuste comercial (%)", -101),
            ("Ajuste comercial (%)", np.inf),
        ]:
            ajustes = self.ajustes(plan)
            ajustes[columna] = valor
            with self.assertRaises(ValueError):
                aplicar_ajustes(plan, ajustes)
        ajustes = self.ajustes(plan)
        with self.assertRaises(ValueError):
            aplicar_ajustes(plan, pd.concat([ajustes, ajustes.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
