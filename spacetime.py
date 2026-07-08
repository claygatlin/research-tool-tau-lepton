import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Superblock Theory Parameters ---
# The theory specifies N_bubble = 1.14 * 10^5. We scale this down 
# slightly for 2D visualization rendering speeds, while keeping density high.
N_BUBBLES = 15000 
GRID_SIZE = 800
FRACTAL_DIMENSION = 2.5       # D ~ 2.5 from the P(k) proportional to k^-2.5 power spectrum
NOISE_STRENGTH_EPSILON = 0.3  # epsilon value (0.1 - 0.3) for the intrinsic roughness

def generate_fractal_noise(size, fractal_dim):
    """
    Generates a 2D stochastic noise field (delta g) with a specific 
    power spectrum P(k) proportional to k^(-D) to represent the 
    bumpy Riemann sphere overlaps.
    """
    # 1. Generate base quantum fluctuations (white noise)
    white_noise = np.random.normal(0, 1, (size, size))
    
    # 2. Transform to frequency domain
    fft_noise = np.fft.fftshift(np.fft.fft2(white_noise))
    
    # 3. Create a radial frequency grid
    x = np.arange(-size // 2, size // 2)
    y = np.arange(-size // 2, size // 2)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    radius[size // 2, size // 2] = 1  # Avoid division by zero at the DC component
    
    # 4. Apply the Superblock power spectrum filter (Amplitude ~ sqrt(P(k)))
    filter_strength = radius ** (-fractal_dim / 2.0)
    fft_filtered = fft_noise * filter_strength
    
    # 5. Transform back to spatial domain
    noise = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    
    # Normalize noise to [-1, 1]
    return (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 2 - 1

def simulate_expanding_foam():
    print("Initializing Automorphic Vacuum...")
    space = np.zeros((GRID_SIZE, GRID_SIZE))
    
    print(f"Nucleating {N_BUBBLES} Planck-Kerr seeds...")
    # Randomly distribute the emergence points (Poisson point process)
    centers_x = np.random.randint(0, GRID_SIZE, N_BUBBLES)
    centers_y = np.random.randint(0, GRID_SIZE, N_BUBBLES)
    
    # Simulate expansion to the percolation threshold (lambda overlap)
    bubble_radius = 12 
    
    print("Expanding Riemann spheres and calculating overlaps...")
    Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    
    # Vectorized distance calculation to build the overlapping cosmic foam
    for x, y in zip(centers_x, centers_y):
        dist_sq = (X - x)**2 + (Y - y)**2
        # Bubble interiors add to the local spacetime density
        space[dist_sq <= bubble_radius**2] += 1
        
    print(f"Applying Higgs quantum noise field (epsilon = {NOISE_STRENGTH_EPSILON})...")
    # Generate the delta_g intrinsic roughness
    delta_g = generate_fractal_noise(GRID_SIZE, FRACTAL_DIMENSION)
    
    # Calculate final density fluctuations: delta_rho / rho ~ epsilon * delta_g
    # We modulate the overlapping bubble field with the fractal noise
    stochastic_spacetime = space * (1 + NOISE_STRENGTH_EPSILON * delta_g)
    
    return stochastic_spacetime

# --- Run Simulation and Visualize ---
density_field = simulate_expanding_foam()

# Plotting the results
plt.figure(figsize=(12, 10))

# Use a colormap that highlights the hot, dense primordial plasma
cmap = plt.get_cmap('magma')
plt.imshow(density_field, cmap=cmap, origin='lower')

plt.colorbar(label='Density Fluctuations ($\delta \\rho / \\rho$)')
plt.title(
    'Superblock Cosmogenesis: Expanding Riemann-Sphere Foam\n'
    f'Fractal Dimension $D={FRACTAL_DIMENSION}$, Noise $\epsilon={NOISE_STRENGTH_EPSILON}$',
    fontsize=14, pad=15
)
plt.axis('off')

# Save and show
plt.tight_layout()
print("Rendering visualization...")
plt.show()
