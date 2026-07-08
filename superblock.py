import numpy as np
import matplotlib.pyplot as plt

# --- Superblock Geometry Parameters [cite: 7232, 7233, 7234, 7235, 7236, 7237, 7238] ---
domains = ['Prime', 'Strange', 'Charm', "Bottom'", "Top'", "Beauty'"]
beta_sq = np.array([0.0, 1/9, 4/9, 8/9, 1/6, 7/12])

# Lorentz factors for the gravitational weight of each domain
gamma = 1 / np.sqrt(1 - beta_sq)

# Volume correction factors (F4 sector vs G2 sector) [cite: 7296]
# Top' and Beauty' are in the F4 sector, which carries an ~0.8 volume factor
v_factors = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.8])

def calculate_dm_ratio(temperature):
    """
    Calculates the Dark Matter to Baryon ratio based on Boltzmann suppression.
    temperature: Effective temperature parameter for the vacuum state.
                 T=1.0 corresponds to the standard e^(-beta^2) suppression.
    """
    # Apply Boltzmann weighting e^(-beta^2 / T) [cite: 7295]
    # For T=1.0, Bottom' probability is e^(-8/9) ≈ 0.41 [cite: 7295]
    probabilities = np.exp(-beta_sq / temperature)
    
    # Calculate effective gamma contributions (Gamma * Prob * Vol_Factor)
    effective_gamma = gamma * probabilities * v_factors
    
    # Baryonic matter is strictly the Prime domain [cite: 7240]
    omega_baryon = effective_gamma[0]
    
    # Dark matter is the sum of the 5 hidden sectors [cite: 7240]
    omega_dm = np.sum(effective_gamma[1:])
    
    return omega_dm / omega_baryon

# --- Simulation Execution ---
# Test a range of "Vacuum Temperatures" to see how the ratio evolves
temperatures = np.linspace(0.5, 3.0, 1000)
ratios = [calculate_dm_ratio(t) for t in temperatures]

# Calculate the baseline theory value at T = 1.0
baseline_ratio = calculate_dm_ratio(1.0)
print(f"Baseline Theory Dark Matter Ratio (T=1.0): {baseline_ratio:.3f}")

# Target observational ratio from Planck 2018
planck_target = 5.3

# Find the specific fine-tuned temperature that matches Planck data
differences = np.abs(np.array(ratios) - planck_target)
best_fit_idx = np.argmin(differences)
fine_tuned_t = temperatures[best_fit_idx]
print(f"Fine-tuned Vacuum Temperature for Planck match: T = {fine_tuned_t:.3f}")

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(temperatures, ratios, label='Theoretical $\Omega_{DM}/\Omega_b$', color='royalblue', linewidth=2)

# Plot reference lines
plt.axhline(y=planck_target, color='crimson', linestyle='--', label=f'Planck 2018 Target (~{planck_target})')
plt.axvline(x=1.0, color='forestgreen', linestyle=':', label='Baseline Vacuum (T=1.0)')

# Highlight the fine-tuned intersection
plt.plot(fine_tuned_t, planck_target, 'ko', markersize=6, label=f'Fine-Tuned Fit (T={fine_tuned_t:.2f})')

plt.title('Superblock Universe: Boltzmann Suppression Fine-Tuning', fontsize=14)
plt.xlabel('Effective Vacuum Temperature Parameter (T)', fontsize=12)
plt.ylabel('Dark Matter to Baryon Ratio ($\Omega_{DM} / \Omega_b$)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
