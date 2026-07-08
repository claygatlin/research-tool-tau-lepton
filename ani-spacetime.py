import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Superblock Theory Parameters ---
# Reduced slightly from the static version for smoother animation rendering
N_BUBBLES = 5000 
GRID_SIZE = 400
FRACTAL_DIMENSION = 2.5       # D ~ 2.5 [cite: 11882]
NOISE_STRENGTH_EPSILON = 0.3  # epsilon value (0.1 - 0.3) [cite: 11881]
MAX_BUBBLE_RADIUS = 12
TOTAL_FRAMES = 60

def generate_fractal_noise(size, fractal_dim):
    """
    Generates a 2D stochastic noise field (delta g) with a specific 
    power spectrum P(k) proportional to k^(-D)[cite: 11886].
    """
    white_noise = np.random.normal(0, 1, (size, size))
    fft_noise = np.fft.fftshift(np.fft.fft2(white_noise))
    x = np.arange(-size // 2, size // 2)
    y = np.arange(-size // 2, size // 2)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    radius[size // 2, size // 2] = 1 
    
    filter_strength = radius ** (-fractal_dim / 2.0)
    fft_filtered = fft_noise * filter_strength
    
    noise = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    return (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 2 - 1

print("Initializing Automorphic Vacuum...")
# Pre-calculate bubble centers and the noise field to keep the animation loop fast
centers_x = np.random.randint(0, GRID_SIZE, N_BUBBLES)
centers_y = np.random.randint(0, GRID_SIZE, N_BUBBLES)
Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]

print("Generating fractal noise field...")
delta_g = generate_fractal_noise(GRID_SIZE, FRACTAL_DIMENSION)

# --- Set up the Figure for Animation ---
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('magma')
im = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap=cmap, origin='lower', vmin=0, vmax=5)
ax.axis('off')

# Add colorbar and title
cbar = plt.colorbar(im, ax=ax, label='Density Fluctuations ($\delta \\rho / \\rho$)')
title = ax.set_title('Superblock Cosmogenesis: Expanding Riemann-Sphere Foam', fontsize=14, pad=15)

def update(frame):
    """
    Animation update function. Runs once per frame.
    """
    space = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Linearly expand the radius of the Riemann spheres over the frames 
    current_radius = (frame / TOTAL_FRAMES) * MAX_BUBBLE_RADIUS
    
    if current_radius > 0:
        for x, y in zip(centers_x, centers_y):
            dist_sq = (X - x)**2 + (Y - y)**2
            space[dist_sq <= current_radius**2] += 1
            
    # Progressively apply the Higgs quantum noise field 
    current_epsilon = NOISE_STRENGTH_EPSILON * (frame / TOTAL_FRAMES)
    stochastic_spacetime = space * (1 + current_epsilon * delta_g)
    
    # Update image data and dynamic color limits to handle the growing density
    im.set_array(stochastic_spacetime)
    
    # Dynamically scale the colorbar so it doesn't wash out as overlaps increase
    max_val = np.max(stochastic_spacetime) if np.max(stochastic_spacetime) > 0 else 1
    im.set_clim(0, max_val * 0.8) 
    
    title.set_text('Superblock Cosmogenesis: Expanding Riemann-Sphere Foam\n'
                   f'Expansion Phase: {frame}/{TOTAL_FRAMES} | Radius: {current_radius:.1f} | '
                   f'Noise $\epsilon$: {current_epsilon:.2f}')
    
    return [im, title]

print("Rendering animation...")
# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=100, blit=True)

# To save the animation as a video file (requires ffmpeg or similar installed):
# ani.save('superblock_big_bang.mp4', writer='ffmpeg', fps=15)

# To save as a GIF (requires imagemagick or pillow installed):
# ani.save('superblock_big_bang.gif', writer='pillow', fps=15)

plt.tight_layout()
plt.show()
