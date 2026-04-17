import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.widgets import Slider # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
from scipy.special import genlaguerre, sph_harm_y # type: ignore
import matplotlib.cm as cm # type: ignore
from matplotlib.colors import Normalize # type: ignore
import math

class AtomicOrbitalVisualizer:
    def __init__(self):
        # Create figure with nice styling - larger size
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.patch.set_facecolor('#0f1419')
        
        # Create main 3D plot (larger, left side)
        self.ax3d = self.fig.add_axes([0.05, 0.30, 0.65, 0.65], projection='3d')
        self.ax3d.set_facecolor('#1a1f2e')
        
        # Create info panel on right side
        self.ax_info = self.fig.add_axes([0.72, 0.30, 0.25, 0.65])
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#1a1f2e')
        
        # Quantum number variables
        self.n = 2
        self.l = 1
        self.m = 1
        self.updating = False  # Flag to prevent recursive updates
        
        # Create sliders positioned below the figure
        ax_n = self.fig.add_axes([0.08, 0.18, 0.55, 0.04])
        ax_l = self.fig.add_axes([0.08, 0.12, 0.55, 0.04])
        ax_m = self.fig.add_axes([0.08, 0.06, 0.55, 0.04])
        
        ax_n.set_facecolor('#252d3d')
        ax_l.set_facecolor('#252d3d')
        ax_m.set_facecolor('#252d3d')
        
        self.slider_n = Slider(ax_n, 'n (Principal)', 1, 5, valinit=2, valstep=1, 
                              color='#FF6B6B', track_color='#384657')
        self.slider_l = Slider(ax_l, 'l (Azimuthal)', 0, 4, valinit=0, valstep=1,
                              color='#4ECDC4', track_color='#384657')
        self.slider_m = Slider(ax_m, 'm (Magnetic)', -4, 4, valinit=0, valstep=1,
                              color='#45B7D1', track_color='#384657')
        
        # Style slider text
        for slider in [self.slider_n, self.slider_l, self.slider_m]:
            plt.setp(slider.label, color='white', fontsize=11, fontweight='bold')
            slider.valtext.set_color('white')
            slider.valtext.set_fontsize(10)
        
        self.slider_n.on_changed(self.on_slider_change)
        self.slider_l.on_changed(self.on_slider_change)
        self.slider_m.on_changed(self.on_slider_change)
        
        # Initial plot
        self.scatter = None
        self.update_orbital()
        
        plt.show()
    
    def validate_quantum_numbers(self, n, l, m):
        """Ensure quantum numbers follow proper constraints"""
        n = max(1, min(int(n), 5))
        l = max(0, min(int(l), n - 1))
        m = max(-l, min(int(m), l))
        return n, l, m
    
    def on_slider_change(self, val):
        """Handle slider changes with proper constraints"""
        if self.updating:
            return
        
        self.updating = True
        try:
            # Get current slider values
            n = int(self.slider_n.val)
            l = int(self.slider_l.val)
            m = int(self.slider_m.val)
            
            # Validate and constrain
            n, l, m = self.validate_quantum_numbers(n, l, m)
            
            # Update sliders to constrained values
            if int(self.slider_n.val) != n:
                self.slider_n.set_val(n)
            if int(self.slider_l.val) != l:
                self.slider_l.set_val(l)
            if int(self.slider_m.val) != m:
                self.slider_m.set_val(m)
            
            # Update the visualization
            self.update_orbital()
        finally:
            self.updating = False
    
    def hydrogen_wavefunction(self, r, theta, phi, n, l, m):
        """
        Calculate hydrogen atom wavefunction probability density
        Uses proper quantum mechanical formulation
        """
        a0 = 1.0  # Bohr radius (atomic units)
        
        # Avoid division by zero and numerical issues
        r = np.atleast_1d(r)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        rho = 2.0 * r / (n * a0)
        rho = np.clip(rho, 1e-6, 150)
        
        # Radial wavefunction
        try:
            L = genlaguerre(n - l - 1, 2 * l + 1)
            prefactor = np.sqrt((2.0 / n) ** 3 * math.factorial(n - l - 1) / 
                              (2 * n * math.factorial(n + l)))
            radial = prefactor * np.exp(-rho / 2.0) * (rho ** l) * L(rho)
            radial = np.real(radial)
        except:
            radial = np.exp(-rho / 2.0) * (rho ** l)
        
        # Angular wavefunction using spherical harmonics
        try:
            angular = sph_harm(m, l, phi, theta) # type: ignore
        except Exception:
            angular = np.ones_like(theta, dtype=np.complex128)

        # Total wavefunction amplitude (angular part oscillates with m)
        psi_total = radial * angular

        # Probability density (always real and positive)
        prob_density = np.abs(psi_total) ** 2
        return np.real(prob_density)
    
    def generate_orbital_data(self, n, l, m, samples=60):
        """Generate 3D point cloud for orbital - sample full 3D space"""
        # Sample all angles uniformly
        theta_samples = int(samples * 0.7)
        phi_samples = int(samples * 0.7)
        
        # Using spherical coordinates: sample all angles, vary radius
        theta = np.linspace(0, np.pi, theta_samples)
        phi = np.linspace(0, 2 * np.pi, phi_samples)
        
        # Radial extent - depends on n
        r_max = 15 + 2 * n
        r = np.linspace(0.01, r_max, samples)
        
        # Create meshgrid for all combinations
        r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian
        x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
        y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
        z = r_grid * np.cos(theta_grid)
        
        # Calculate wavefunction values
        psi_values = self.hydrogen_wavefunction(r_grid, theta_grid, phi_grid, n, l, m)
        
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        psi_flat = psi_values.flatten()
        
        # Filter by threshold to show orbital shape clearly
        threshold = np.max(psi_flat) * 0.15  # Adjust for visibility
        mask = psi_flat > threshold
        
        x_filtered = x_flat[mask]
        y_filtered = y_flat[mask]
        z_filtered = z_flat[mask]
        colors_filtered = psi_flat[mask]
        
        # If too many points, subsample
        if len(x_filtered) > 15000:
            indices = np.random.choice(len(x_filtered), 15000, replace=False)
            x_filtered = x_filtered[indices]
            y_filtered = y_filtered[indices]
            z_filtered = z_filtered[indices]
            colors_filtered = colors_filtered[indices]
        
        return x_filtered, y_filtered, z_filtered, colors_filtered
    
    def update_orbital(self):
        """Update the orbital visualization"""
        # Get and validate slider values
        n = int(self.slider_n.val)
        l = int(self.slider_l.val)
        m = int(self.slider_m.val)
        
        n, l, m = self.validate_quantum_numbers(n, l, m)
        
        # Only update if values actually changed
        if (n, l, m) == (self.n, self.l, self.m) and hasattr(self, '_plotted'):
            return
        
        self.n, self.l, self.m = n, l, m
        self._plotted = True
        
        # Clear 3D plot
        self.ax3d.clear()
        self.ax3d.set_facecolor('#1a1f2e')
        
        # Generate orbital data
        x, y, z, colors = self.generate_orbital_data(n, l, m, samples=65)
        
        # Plot with better colormap
        if len(x) > 100:
            scatter = self.ax3d.scatter(x, y, z, c=colors, cmap='twilight', 
                                       s=15, alpha=0.5, edgecolors='none',
                                       norm=Normalize(vmin=np.min(colors), vmax=np.max(colors)))
        
        # Set equal aspect and limits
        scale = 15 + 2 * n
        self.ax3d.set_xlim(-scale, scale)
        self.ax3d.set_ylim(-scale, scale)
        self.ax3d.set_zlim(-scale, scale)
        
        # Customize grid and panes
        self.ax3d.xaxis.pane.fill = False
        self.ax3d.yaxis.pane.fill = False
        self.ax3d.zaxis.pane.fill = False
        
        self.ax3d.xaxis.pane.set_edgecolor('#384657')
        self.ax3d.yaxis.pane.set_edgecolor('#384657')
        self.ax3d.zaxis.pane.set_edgecolor('#384657')
        
        self.ax3d.xaxis.pane.set_linewidth(0.5)
        self.ax3d.yaxis.pane.set_linewidth(0.5)
        self.ax3d.zaxis.pane.set_linewidth(0.5)
        
        # Remove all axis labels and ticks
        self.ax3d.set_xlabel('')
        self.ax3d.set_ylabel('')
        self.ax3d.set_zlabel('')
        
        # Hide tick labels
        self.ax3d.set_xticklabels([])
        self.ax3d.set_yticklabels([])
        self.ax3d.set_zticklabels([])
        
        # Remove grids
        self.ax3d.grid(False)
        
        # Update info panel with constraints and orbital info
        self.update_info_panel()
        
        self.fig.canvas.draw_idle()
    
    def update_info_panel(self):
        """Update information display on right side"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#1a1f2e')
        
        orbital_name = self.get_orbital_name()
        orbital_desc = self.get_orbital_description()
        electron_count = self.get_electron_count()
        
        info_text = f"""━━━━━━━━━━━━━━━
QUANTUM NUMBERS
━━━━━━━━━━━━━━━

n = {self.n}
l = {self.l}
m = {self.m}

━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━

n: 1 to 5
l: 0 to {self.n-1}
  (l < n)
m: {-self.l} to {self.l}
  (|m| ≤ l)

━━━━━━━━━━━━━━━
ORBITAL INFO
━━━━━━━━━━━━━━━

Orbital: {orbital_name}
Shape: {orbital_desc}

Max Electrons: {electron_count}

Radial Nodes: {self.n - self.l - 1}
Angular Nodes: {self.l}

Energy Level: n={self.n}
"""
        
        self.ax_info.text(0.05, 0.95, info_text, fontsize=9,
                         family='monospace', fontweight='bold',
                         transform=self.ax_info.transAxes,
                         verticalalignment='top', horizontalalignment='left',
                         color='#00D9FF',
                         bbox=dict(boxstyle='round,pad=0.8',
                         facecolor='#252d3d', edgecolor='#45B7D1', linewidth=1.5, alpha=0.9))
    
    def get_orbital_name(self):
        """Get orbital name (like 1s, 2p, 3d, etc.)"""
        orbital_symbols = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        symbol = orbital_symbols.get(self.l, '?')
        return f"{self.n}{symbol}"
    
    def get_orbital_description(self):
        """Get orbital shape description"""
        descriptions = {
            's': "Spherical",
            'p': "Dumbbell",
            'd': "Cloverleaf",
            'f': "Complex",
        }
        symbol = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}.get(self.l, '?')
        return descriptions.get(symbol, "Complex")
    
    def get_electron_count(self):
        """Get max electrons in this orbital"""
        return 2 * (2 * self.l + 1)

if __name__ == "__main__":
    visualizer = AtomicOrbitalVisualizer()