## 2025-03-08 - Avoid repeated float casting and tensor subtraction in tight training loops
**Learning:** In PyTorch tight training loops, avoid recomputing identical float castings and subtractions (e.g., `acts_mb.float() - x_hat.float()`). Computing it once and saving it to a variable, then using `.detach()` where gradients are not needed, yields significant performance gains.
**Action:** Always look for repeated expensive operations (like dtype casting and element-wise arithmetic) inside the innermost loops and cache their results to reuse them.
