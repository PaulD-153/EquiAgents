import numpy as np
import matplotlib.pyplot as plt

# Violation range
violations = np.linspace(-1.0, 1.0, 500)
k = 5  # Curvature parameter

# Compute delta using the exponential update rule
delta = np.sign(violations) * (np.exp(k * np.abs(violations)) - 1)
# Fix label formatting to avoid LaTeX parsing issues
plt.figure(figsize=(8, 5))
plt.plot(violations, delta, label='Delta update signal')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Fairness violation")
plt.ylabel("Update signal (Delta)")
plt.title("Exponential Update Signal vs. Fairness Violation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()