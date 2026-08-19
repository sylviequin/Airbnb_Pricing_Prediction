# Data

The raw CSVs (`train.csv`, `test.csv`) are not committed to this repo — this folder is a placeholder.

## Source

TODO — confirm and link the exact source. The column schema (`neighbourhood_cleansed`, `host_since`, `review_scores_rating`, `estimated_revenue_l365d`, etc.) matches the format published by [Inside Airbnb](http://insideairbnb.com/get-the-data/), which publishes free scrapes of Airbnb listings per city for non-commercial research/analysis use, with attribution requested. If this data came from a Kaggle competition/dataset built on that scrape instead, link the Kaggle page here.

## Getting the data

1. Download the listings CSV for the relevant city/date from the source above.
2. Place it in this folder as `train.csv` (and `test.csv` if you're using a separate holdout file).

## Columns of note

- `price` — target variable
- `neighbourhood_cleansed`, `latitude`, `longitude` — location
- `property_type`, `room_type`, `accommodates`, `bedrooms`, `bathrooms` — property attributes
- `host_since`, `host_is_superhost`, `host_response_rate` — host attributes
- `review_scores_rating` and sub-scores, `number_of_reviews`, `reviews_per_month` — review signals
- `availability_30/60/90/365`, `minimum_nights`, `maximum_nights` — availability/booking terms
