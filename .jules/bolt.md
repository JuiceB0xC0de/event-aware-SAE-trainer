## 2025-03-08 - Avoid repeated float casting and tensor subtraction in tight training loops
**Learning:** In PyTorch tight training loops, avoid recomputing identical float castings and subtractions (e.g., `acts_mb.float() - x_hat.float()`). Computing it once and saving it to a variable, then using `.detach()` where gradients are not needed, yields significant performance gains.
**Action:** Always look for repeated expensive operations (like dtype casting and element-wise arithmetic) inside the innermost loops and cache their results to reuse them.

## 2025-03-08 - Avoid redundant forward passes by caching residuals
**Learning:** During steps requiring an error buffer for resampling, the naive approach re-computed a full encoder-decoder forward pass on the entire batch to find high-error tokens. By accumulating `residual_float.pow(2).sum(dim=-1)` inside the microbatch training loop where `residual_float` is already computed, we entirely avoid an expensive, redundant O(batch_tokens * D_IN * n_features) forward pass.
**Action:** Always check if a metric (like `per_token_err`) can be extracted from intermediate variables already computed during the primary training forward pass.
