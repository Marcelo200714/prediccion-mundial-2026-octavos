"""
Configuracion central del proyecto: rutas, constantes del modelo y
parametros extraidos de 'Cerebro_Del_Modelo.md'.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"

MATCHES_CSV = DATA_DIR / "matches.csv"
TEAMS_CSV = DATA_DIR / "teams.csv"
MATCH_TEAM_STATS_CSV = DATA_DIR / "match_team_stats.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"
SQUADS_CSV = DATA_DIR / "squads_and_players.csv"

# Ventanas de medias moviles ("forma reciente"), sec. 2.2 del manual
VENTANAS = [2, 5]

# Umbral de goleada para sobreponderar muestras de entrenamiento, sec. 3.1
UMBRAL_GOLEADA = 3
PESO_GOLEADA = 1.5

# Decaimiento exponencial de recencia, sec. 3.5 (DECAY_RECENCIA=0.0025 en el manual;
# aqui se reescala porque nuestro historico llega a 150 anios, no unos pocos)
DECAY_RECENCIA = 0.0006

# Peso por confederacion (adaptado de "pesos_continente", sec. 2.5)
PESOS_CONFEDERACION = {
    "UEFA": 1.00,      # Europa
    "CONMEBOL": 0.95,  # Sudamerica
    "CONCACAF": 0.75,  # Norteamerica
    "CAF": 0.60,       # Africa
    "AFC": 0.70,       # Asia
    "OFC": 0.50,       # Oceania
}

# Normalizacion de nombres: teams.csv (Mundial 2026) -> results.csv / shootouts.csv (historico)
MAPEO_NOMBRES = {
    "Czechia": "Czech Republic",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
}

TWEEDIE_VARIANCE_POWER = 1.5
RANDOM_STATE = 42
N_SIMULACIONES_MONTECARLO = 10_000

MODELS_DIR.mkdir(parents=True, exist_ok=True)
