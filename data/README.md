# Data Dictionary – Retailrocket E-commerce Dataset

- **Source:** [Retailrocket Recommender System Dataset (Kaggle)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **Raw Files in `data/raw/`:**
  - `events.csv` (`2,756,101` rows × `5` columns, `90 MB`, covering `2015-05-03` to `2015-09-18`)
  - `category_tree.csv` (`1,669` rows × `2` columns: `categoryid`, `parentid`, with `25` root categories)
  - `item_properties_part1.csv` (`10,999,999` rows × `4` columns, `462 MB`)
  - `item_properties_part2.csv` (`9,275,903` rows × `4` columns, `390 MB`)
- **Processed Modeling File:** `data/processed/clean.csv` (`30,000` leakage-safe early-session snapshots × `13` columns)
- **Processed Customer Profiles:** `data/processed/customer_profiles.csv` (`25,000` engaged customers × `11` columns)

## 1. Raw Clickstream Columns (`data/raw/events.csv`)

| Column | Type | Description |
|---|---|---|
| `timestamp` | integer (epoch ms) | Unix timestamp of the interaction in milliseconds (UTC). |
| `visitorid` | integer | Anonymous visitor identifier (`1,407,580` unique visitors). |
| `event` | category | Interaction type: `view` (`2,664,312`), `addtocart` (`69,332`), or `transaction` (`22,457`). |
| `itemid` | integer | Unique product identifier (`235,061` distinct items). |
| `transactionid` | integer (nullable) | Order identifier; present only when `event == "transaction"`. |

## 2. Processed Purchase Prediction Columns (`data/processed/clean.csv`)

| Column | Type | Role | Description |
|---|---|---|---|
| `session_id` | integer | Dropped ID | Engineered session identifier (30-minute inactivity cutoff). Excluded from model inputs. |
| `visitorid` | integer | Dropped ID | Visitor identifier. Excluded from model inputs. |
| `session_start_ts` | integer | Dropped ID | Session start timestamp (ms). Excluded from model inputs. |
| `views_count` | integer | Input feature | Number of product views in the first 3 non-transaction events (`0–3`). |
| `unique_items_viewed` | integer | Input feature | Distinct products interacted with in the first 3 events (`1–3`). |
| `add_to_cart_count` | integer | Input feature | Number of add-to-cart events in the first 3 events (`0–3`). |
| `elapsed_seconds` | float | Input feature | Seconds elapsed between the 1st and 3rd event in the session. |
| `ends_with_cart` | category (`Yes`/`No`) | Input feature | Whether the 3rd event was an `addtocart` action. |
| `start_hour` | integer (`0–23`) | Input feature | Hour of day (UTC) when the session started. |
| `is_weekend` | category (`Yes`/`No`) | Input feature | Whether the session occurred on Saturday or Sunday. |
| `prior_sessions` | integer | Input feature | Number of strictly earlier sessions for this visitor. |
| `prior_purchases` | integer | Input feature | Number of transactions completed by this visitor in strictly earlier sessions. |
| `will_purchase` | target (`Yes`/`No`) | **Target** | `Yes` if a transaction occurs strictly after the 3rd event in the same session, else `No`. |

## 3. Customer Segmentation Columns (`data/processed/customer_profiles.csv`)

| Column | Type | Role | Description |
|---|---|---|---|
| `visitorid` | integer | Identifier | Customer ID (excluded from K-Means inputs). |
| `total_sessions` | integer | K-Means feature | Total browsing sessions recorded for the customer. |
| `total_views` | integer | K-Means feature | Total product view events. |
| `unique_items_viewed` | integer | K-Means feature | Total distinct products viewed across sessions. |
| `total_add_to_cart` | integer | K-Means feature | Total add-to-cart events. |
| `avg_session_duration_sec` | float | K-Means feature | Average session duration in seconds. |
| `avg_items_per_session` | float | K-Means feature | Average distinct products explored per session. |
| `cart_abandonment_rate` | float (`0–1`) | K-Means feature | Share of cart sessions that did not end in a transaction. |
| `recency_days` | float | K-Means feature | Days between the customer's last activity and the end of the dataset. |
| `total_transactions` | integer | Post-cluster eval | Total purchase events (used only to compare segments after clustering). |
| `converted` | category (`Yes`/`No`) | Post-cluster eval | Whether the customer ever completed a purchase. |
