# %%
from regime_partitioning.datasets import fx_datasets

df_fx = fx_datasets["EURUSD"]["df_fx"]

# %%
df_fx["rate_diff_2y"]

# %% [markdown]
"""
datetime
2020-01-30   -1.885735
2020-01-31   -1.756128
2020-02-03   -1.403285
2020-02-04   -1.627197
2020-02-05   -0.746616
                ...   
2024-12-20   -1.104609
2024-12-23   -0.856455
2024-12-24   -1.644762
2024-12-27   -1.019297
2024-12-30   -1.250853
Name: rate_diff_2y, Length: 1215, dtype: float64
"""
# %%

from regime_partitioning.processing import pelt_changepoints, fit_2state_hmm
import numpy as np

# Choose simple penalties. Tune as needed.
pen_rate = 3.0 * np.log(len(df_fx))
pen_rv = 3.0 * np.log(len(df_fx))

cp_rate = pelt_changepoints(df_fx["rate_diff_2y"], penalty=pen_rate, min_size=20)
cp_rv = pelt_changepoints(df_fx["rv_20d"], penalty=pen_rv, min_size=20)

# cp_rate and cp_rv are lists of timestamps where a new regime starts after that date.

hmm_model, scaler, hmm_df = fit_2state_hmm(df_fx, cols=("ret", "rv_20d"))

# -------------------------
# 3) Example outputs
# -------------------------
print("PELT changepoints on 2y rate diff:", cp_rate)
print("PELT changepoints on 20d RV:", cp_rv)
print(hmm_df[["state", "p_state0", "p_state1", "regime"]].tail())

# If you also want segment labels aligned to full df:
df_fx_out = df_fx.join(hmm_df[["state", "p_state0", "p_state1", "regime"]], how="left")

# %% [markdown]
"""
PELT changepoints on 2y rate diff: [Timestamp('2020-03-26 00:00:00'), Timestamp('2022-02-28 00:00:00')]
PELT changepoints on 20d RV: []
            state  p_state0  p_state1    regime
datetime                                       
2024-12-20      0  0.978699  0.021301  risk_off
2024-12-23      0  0.992199  0.007801  risk_off
2024-12-24      0  0.996911  0.003089  risk_off
2024-12-27      0  0.998360  0.001640  risk_off
2024-12-30      0  0.997570  0.002430  risk_off
"""


# %%

print("PELT changepoints on 2y rate diff:", cp_rate)
print("PELT changepoints on 20d RV:", cp_rv)
print(hmm_df[["state", "p_state0", "p_state1", "regime"]].tail())

# If you also want segment labels aligned to full df:
df_fx_out = df_fx.join(hmm_df[["state", "p_state0", "p_state1", "regime"]], how="left")

# %% [markdown]

"""
PELT changepoints on 2y rate diff: [Timestamp('2020-03-26 00:00:00'), Timestamp('2022-02-28 00:00:00')]
PELT changepoints on 20d RV: []
            state  p_state0  p_state1    regime
datetime                                       
2024-12-20      0  0.978699  0.021301  risk_off
2024-12-23      0  0.992199  0.007801  risk_off
2024-12-24      0  0.996911  0.003089  risk_off
2024-12-27      0  0.998360  0.001640  risk_off
2024-12-30      0  0.997570  0.002430  risk_off
"""
