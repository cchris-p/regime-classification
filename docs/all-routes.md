Here’s the output formatted cleanly in Markdown for readability:

---

# 🚀 TradesSurge Analytics – FastAPI Routes

---

## 📁 COT Routes

* **GET** `/api/cot/signal_matrix`

---

## 📁 DOCS Routes

* **GET** `/docs`
* **GET** `/docs/oauth2-redirect`
* **GET** `/openapi.json`
* **GET** `/redoc`

---

## 📁 ECONOMICS Routes

* **GET** `/api/economics/{currency}/cpi/{granularity}`
* **GET** `/api/economics/{currency}/gdp/{granularity}`
* **GET** `/api/economics/{currency}/interest_rates/{granularity}`
* **GET** `/api/economics/{currency}/unemployment/{granularity}`

---

## 📁 FOREX Routes

* **GET** `/api/forex/historical-data/eodhd/`
* **GET** `/api/forex/historical-data/firstrate/`
* **GET** `/api/forex/historical-data/ft5/`

---

## 📁 FUTURES Routes

* **GET** `/api/futures/historical-data/eodhd/`
* **GET** `/api/futures/historical-data/firstrate/`
* **GET** `/api/futures/historical-data/firstrate/`
* **GET** `/api/futures/historical-data/ft5/`

---

## 📁 SHOW-ROUTES Routes

* **GET** `/api/show-routes`

---

### 📊 Total Routes: **17**

---

# 🔍 Economics Endpoint Tests

| Endpoint                                    | Status | Notes                             |
| ------------------------------------------- | ------ | --------------------------------- |
| `/api/economics/usd/unemployment/quarterly` | ⚠️ 503 | Unable to connect to external API |
| `/api/economics/eur/cpi/monthly`            | ✅ 200  | OK                                |
| `/api/economics/aud/gdp/annual`             | ⚠️ 503 | Unable to connect to external API |
| `/api/economics/gbp/interest_rates/monthly` | ⚠️ 503 | Unable to connect to external API |

---

Do you want me to also rewrite your `routes.sh` script so that it always prints in this table format (Django-style `show_urls`), instead of the raw log output?
