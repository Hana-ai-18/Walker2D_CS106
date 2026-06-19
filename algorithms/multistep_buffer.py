"""
Multi-step buffer note:
SB3 >= 2.0 supports n-step returns natively via the `n_steps` parameter
in TD3/SAC constructors. The buffer computes discounted n-step returns
and exposes replay_data.discounts = gamma^n for the bootstrapping.

This file is kept for documentation/compatibility.
For MTD3/MSAC we use the built-in SB3 n-step support.
"""
