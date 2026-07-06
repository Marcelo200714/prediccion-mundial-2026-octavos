"""
Punto de entrada unico del proyecto. Ejecuta la tuberia completa:
carga -> ingenieria de variables -> modelado XGBoost en cascada ->
prediccion de Octavos de Final -> visualizacion final.

Uso:
    python src/main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from config import MODELS_DIR, OUTPUT_DIR
import data_loader as dl
import feature_engineering as fe
import modeling as mdl
import predict as pr
import visualize as viz


def main():
    print("=" * 70)
    print("PROYECTO PREDICTIVO MUNDIAL 2026 -- Octavos de Final -> Cuartos")
    print("=" * 70)

    print("\n[1/5] Cargando fuentes crudas (teams, results, matches, squads)...")
    teams = dl.cargar_equipos()
    valor_mercado = dl.cargar_valor_mercado()
    historico = dl.cargar_historico_resultados()
    shootouts = dl.cargar_shootouts()
    historico = dl.resolver_ganador_penaltis(historico, shootouts)
    matches_2026 = dl.construir_matches_2026_enriquecido(teams)
    print(f"      {len(historico):,} partidos historicos (1872-2026) | {len(teams)} selecciones Mundial 2026")

    print("\n[2/5] Ingenieria de variables (medias moviles, H2H, Elo, diff_*)...")
    df_modelo = fe.construir_dataset_modelo(historico, matches_2026, teams, valor_mercado)
    print(f"      Dataset final: {len(df_modelo):,} partidos entre selecciones del Mundial 2026")

    pendientes_mask = df_modelo["home_score"].isna() & (df_modelo["stage_id"] == 3)
    decididos_mask = df_modelo["home_score"].notna() & (df_modelo["stage_id"] == 3)
    df_train = df_modelo[df_modelo["home_score"].notna()].reset_index(drop=True)
    df_pendientes = df_modelo[pendientes_mask].reset_index(drop=True)
    df_decididos = df_modelo[decididos_mask].reset_index(drop=True)
    print(f"      Entrenamiento: {len(df_train):,} partidos | Octavos ya jugados: {len(df_decididos)} | "
          f"Octavos por predecir: {len(df_pendientes)}")

    print("\n[3/5] Entrenando cascada XGBoost (Piso 1: goles Tweedie | Piso 2: 1X2 calibrado)...")
    modelo = mdl.entrenar_pipeline(df_train)
    joblib.dump(modelo, MODELS_DIR / "modelo_cascada_octavos.pkl")
    print(f"      Modelos guardados en {MODELS_DIR}")

    print("\n[4/5] Prediciendo TODOS los partidos de Octavos con el modelo (espejo + Monte Carlo), "
          "incluidos los ya jugados de Marruecos y Francia...")
    df_todos_octavos = pd.concat([df_pendientes, df_decididos], ignore_index=True)
    resultados = pr.predecir_partidos_pendientes(df_todos_octavos, modelo)

    for _, r in resultados.sort_values("match_id").iterrows():
        estado = "(real)" if r["ya_jugado"] else f"({r['confianza']*100:.0f}% conf.)"
        print(f"      {r['equipo_local']:<24} vs {r['equipo_visitante']:<24} -> "
              f"avanza {r['ganador_predicho']:<20} {estado}")

    resultados.to_csv(OUTPUT_DIR / "predicciones_octavos.csv", index=False)

    print("\n[5/5] Generando 'clasificados_octavos.png'...")
    ruta_img = OUTPUT_DIR / "clasificados_octavos.png"
    viz.generar_imagen(resultados, ruta_img)
    print(f"      Imagen guardada en {ruta_img}")

    print("\n" + "=" * 70)
    print("LOS 8 EQUIPOS QUE CLASIFICAN A CUARTOS DE FINAL:")
    for equipo in resultados.sort_values("match_id")["ganador_predicho"]:
        print(f"   -> {equipo}")
    print("=" * 70)


if __name__ == "__main__":
    main()
