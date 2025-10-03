PELT = **Pruned Exact Linear Time**.

It is an algorithm for **change-point detection** in time series. Purpose: split a sequence into segments where statistical properties (like mean or variance) are stable.

Key points:

* **Cost function**: measures how well each segment fits a model (commonly mean/variance shift).
* **Penalty**: discourages too many change-points. Larger penalty → fewer regimes.
* **Pruning**: eliminates candidate change-points that cannot be optimal, making the method run in **O(n)** average time (linear in data length).

Usage in regime detection:

* Input: series like `rate_diff_2y`, inflation differentials, realized volatility.
* Output: estimated breakpoints where structural shifts occur.
* Advantages: efficient, consistent, works on large samples; unlike binary segmentation it does not miss breaks when multiple shifts are close.
* Implementation: Python `ruptures` library (`ruptures.Pelt(model="rbf").fit(series).predict(pen=β)`), or direct C++ implementations using dynamic programming with pruning.

Would you like me to write out a **minimal Python snippet** with `ruptures` that applies PELT to `rate_diff_2y` and returns regime boundaries, ready to plug into your pipeline?
