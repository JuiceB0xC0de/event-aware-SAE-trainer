## 2025-03-08 - Avoid repeated float casting and tensor subtraction in tight training loops
**Learning:** In PyTorch tight training loops, avoid recomputing identical float castings and subtractions (e.g., `acts_mb.float() - x_hat.float()`). Computing it once and saving it to a variable, then using `.detach()` where gradients are not needed, yields significant performance gains.
**Action:** Always look for repeated expensive operations (like dtype casting and element-wise arithmetic) inside the innermost loops and cache their results to reuse them.

## 2025-03-08 - Avoid redundant full-batch forward passes during dead feature resampling
**Learning:** The previous implementation performed a redundant full-batch forward pass to compute per-token errors for dead feature resampling. This caused large peak VRAM spikes since it bypassed the microbatching logic. Re-using the detached residuals from the microbatch pass is just as effective and significantly reduces both VRAM consumption and redundant computation.
**Action:** Avoid full-batch evaluation where detached residuals from microbatching can be re-used.
