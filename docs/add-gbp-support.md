`GBPUSD` can be added to `fx_datasets`, but the dataset build path also requires `rate_diff_2y` and `cpi_diff_core` features. `cpi_diff_core` for GBPUSD already exists, but there is currently no UK/GB 2Y yield file in `data/yields/clean` (only US/EU/JP/AU/NZ/CA), so `GBPUSD_rate_diff_2y` cannot be created from real data yet.

Recommended implementation options:
1. **Proper data path (preferred):** add UK 2Y yield source file(s), compute `GBPUSD_rate_diff_2y`, then add `GBPUSD_dataset` and registry entry.
2. **Temporary proxy path:** add a synthetic/proxy GBP 2Y series so `GBPUSD` exports run immediately, with the understanding that regime labels are approximate.

If this direction is acceptable, please **toggle to Act mode** and confirm which option to implement.