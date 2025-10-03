Short answer: only if they increase **signal** without adding noise.

* **2-year yield for `rate_diff_2y`:**

  * Best: **daily** 2y yields (per country), then forward-fill to trading days. Improves PELT sensitivity and HMM inputs.
  * Acceptable: **monthly** 2y or **monthly policy rate proxy**. Works, but change-points land coarsely.
  * Avoid: **quarterly/annual** for this feature. Too coarse for your plan.

* **CPI core YoY for `cpi_diff_core`:**

  * Use **monthly**. That’s the publication cadence. Do not upsample beyond forward-filling to daily.

* **VIX, FX returns, `rv_20d`:**

  * Use **daily**. Compute `rv_20d` from daily log returns.

* **Event anchors:**

  * Keep as **exact dates** (decision days, YCC changes). Do not resample.

Rule of thumb for this plan:

* Pick the **highest native frequency** that is reliable for each series.
* Forward-fill to the **daily** FX calendar.
* Run **PELT** on daily `rate_diff_2y` and daily `rv_20d`.
* Run the **2-state HMM** on daily `[ret, rv]`.
* Quarterly/annual macro is fine for extended models, not for this minimal setup.
