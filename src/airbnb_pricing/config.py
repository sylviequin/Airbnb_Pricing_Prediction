"""Central place for paths, constants, feature lists, and hyperparameters, not the hard code.

Ported from the group's original notebook (`notebook/airbnb_pricing_prediction.ipynb`,
BUSA3020 Kaggle assignment, team BUSA3020_Dataholic) so train.py, predict.py, and the
notebook all agree on the same settings instead of redefining lists in every cell.
"""

from pathlib import Path

import pandas as pd

# Paths: import the local path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_TRAIN_PATH = DATA_DIR / "train.csv"
RAW_TEST_PATH = DATA_DIR / "test.csv"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Modeling 
TARGET_COLUMN = "price"
RANDOM_STATE = 42
CV_FOLDS = 5

        # Columns dropped outright: identifiers, free text, or redundant with another column.
        # `estimated_revenue_l365d` is dropped separately in data.py because it's derived from price
        # on the platform side and would leak the target.
COLUMNS_TO_DROP = [
    "description",
    "neighborhood_overview",
    "host_name",
    "host_id",
    "host_about",
    "host_location",
    "host_neighbourhood",
    "host_verifications",
    "bathrooms_text",
]

        # Feature engineering 
        # Snapshot date used to compute "days since X" features — fixed so the pipeline
        # is reproducible rather than depending on the day it happens to run.
SNAPSHOT_DATE = pd.Timestamp("2025-12-01")
CBD_LAT, CBD_LON = -27.4698, 153.0251  # Brisbane CBD

        # Columns superseded by an engineered feature (e.g. availability_365 -> availability_rate)
        # or no longer needed once their derived features are built.
DROP_AFTER_ENGINEERING = [
    "availability_365",
    "latitude",
    "longitude",
    "host_since",
    "last_review",
]

        # Geo clustering
N_GEO_CLUSTERS = 5

        # Rare-category collapsing: nominal columns with more than TOP_N_CATEGORIES unique
        # values get everything outside the top N lumped into "other" (fit on train only).
CAT_NOMINAL = ["room_type", "property_type", "neighbourhood_cleansed", "geo_cluster"]
TOP_N_CATEGORIES = 5

        # Ordinal encoding for host_response_time — faster response = lower value.
RESPONSE_TIME_ORDER = {
    "within an hour": 0,
    "within a few hours": 1,
    "within a day": 2,
    "a few days or more": 3,
    "unknown": 4,
}

BOOLEAN_COLUMNS = ["host_is_superhost", "instant_bookable", "host_identity_verified"]

REVIEW_SCORE_FEATURES = [
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_accuracy",
    "review_scores_communication",
    "review_scores_checkin",
    "review_scores_value",
]

BASE_NUM_FEATURES = [
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "estimated_occupancy_l365d",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "reviews_per_month",
    *REVIEW_SCORE_FEATURES,
]

ENG_NUM_FEATURES = [
    "amenities_count",
    "host_experience_days",
    "days_since_last_review",
    "is_commercial_host",
    "beds_per_person",
    "bath_to_bedroom_ratio",
    "accommodates_per_bedroom",
    "availability_rate",
    "dist_to_cbd",
]

CAT_FEATURES = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_response_time",
    "geo_cluster",
]

        # Columns imputed with the train-set median (fit on train only, to avoid leakage).
MEDIAN_IMPUTE_COLUMNS = [
    "bathrooms",
    "beds",
    "bedrooms",
    "host_listings_count",
    "host_response_rate",
    "host_acceptance_rate",
    "host_experience_days",
    "beds_per_person",
    "bath_to_bedroom_ratio",
    "accommodates_per_bedroom",
    *REVIEW_SCORE_FEATURES,
]

        # Amenity keyword -> regex, matched against the raw `amenities` text column.
        # Order matters for a couple of overlapping patterns (e.g. dryer vs hair dryer).
AMENITY_MAP = {
    "has_wifi": r"wi-?fi",
    "has_heating": r"heating",
    "has_essentials": r"essentials",
    "has_kitchen": r"kitchen",
    "has_smoke_alarm": r"smoke alarm",
    "has_air_conditioning": r"air conditioning",
    "has_hangers": r"hangers",
    "has_workspace": r"workspace",
    "has_coffee_maker": r"coffee maker",
    "has_hot_water_kettle": r"hot water kettle",
    "has_hair_dryer": r"hair dryer",
    "has_iron": r"\biron\b",
    "has_tv": r"\btv\b|hdtv|television",
    "has_hot_water": r"hot water(?! kettle)",
    "has_washer": r"\bwasher\b",
    "has_dryer": r"(?<!hair )dryer",
    "has_first_aid_kit": r"first aid kit",
    "has_fire_extinguisher": r"fire extinguisher",
    "has_refrigerator": r"refrigerator",
    "has_pool": r"\bpool\b",
    "has_gym": r"gym|fitness|exercise equipment",
    "has_pets_allowed": r"pets allowed",
    "has_free_parking": r'free[^"]*parking',
    "has_paid_parking": r'paid[^"]*parking',
    "has_self_checkin": r"self check-?in",
}

        # Listing-title keyword -> regex, matched against the raw `name` text column.
NAME_KEYWORDS = {
    "name_luxury": r"luxury|luxurious",
    "name_pool": r"\bpool\b",
    "name_view": r"\bview\b|views",
    "name_studio": r"\bstudio\b",
    "name_cbd": r"\bcbd\b|city centre|city center",
    "name_beach": r"\bbeach\b|beachside|beachfront",
    "name_ocean": r"\bocean\b",
    "name_penthouse": r"\bpenthouse\b",
    "name_modern": r"modern|contemporary",
}
