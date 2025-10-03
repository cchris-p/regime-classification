# Differential Series Utilities

## `calculate_interest_rate_difference`

Compute a monthly policy-rate differential series for a currency pair.

**Signature**

```python
calculate_interest_rate_difference(economic_data, base_currency, quote_currency, frequency="monthly") -> pd.Series
```

**Parameters**

* `economic_data` dict from `get_economic_data()`. Must contain `["<CCY>"]["interest_rates"][frequency]`.
* `base_currency` str. e.g., `"EUR"`.
* `quote_currency` str. e.g., `"USD"`.
* `frequency` str. Default `"monthly"`.

**Returns**

* `pd.Series` indexed by period. Values are `base − quote`. Name: `"{BASE}_{QUOTE}_diff"`.

**Notes**

* Expects records with `{"period": <str>, "value": <float>}`.
* Aligns by datetime index. Missing side uses `fill_value=0` on subtraction.

**Example**

```python
diff = calculate_interest_rate_difference(econ, "EUR", "USD")
```

## `calculate_dei`

Compute a Differential Economic Indicator (DEI) for a pair, enforcing IMF/ECB period support.

**Signature**

```python
calculate_dei(base, quote, indicator_type, period_type, economic_data) -> pd.DataFrame
```

**Parameters**

* `base` str. Base currency code.
* `quote` str. Quote currency code.
* `indicator_type` str. One of `["interest_rates","gdp","cpi","unemployment"]`.
* `period_type` str. One of `["monthly","quarterly","annual","yearly"]` depending on source rules.
* `economic_data` dict from `get_economic_data()`.

**Returns**

* `pd.DataFrame` with columns:

  * `Time` (datetime)
  * `"{BASE}/{QUOTE} {indicator_type} {period_type} DEI"` = `base − quote`

**Validation rules**

* For `EUR` (ECB):

  * `interest_rates`: monthly
  * `gdp`: quarterly
  * `cpi`: monthly
  * `unemployment`: quarterly
* For non-`EUR` (IMF):

  * `interest_rates`: monthly
  * `gdp`: yearly, quarterly
  * `cpi`: annual, monthly, quarterly
  * `unemployment`: annual, monthly, quarterly

**Notes**

* Inner-joins on overlapping `period`s only.
* Coerces `period` to datetime.

**Example**

```python
dei = calculate_dei("USD","EUR","cpi","monthly",econ)
```
