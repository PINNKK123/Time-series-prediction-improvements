"""
Variational Mode Decomposition (VMD) with Bayesian optimization.
Extended parameter search space for better decomposition quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Import VMD - note: vmdpy may not be available, so we'll provide a fallback
try:
    from vmdpy import VMD
    VMD_AVAILABLE = True
except ImportError:
    VMD_AVAILABLE = False
    print("Warning: vmdpy not available. Using simplified VMD implementation.")


class VMDDecomposer:
    """
    VMD decomposer with extended parameter space and quality checks.
    """
    
    # Quality score weights
    ORTHOGONALITY_WEIGHT = 0.5
    SPARSITY_WEIGHT = 0.3
    
    def __init__(self, k_range=(3, 8), alpha_range=(1e2, 1e4), tau=0.0, DC=0, init=1, tol=1e-7):
        """
        Initialize VMD decomposer.
        
        Args:
            k_range: Range for number of modes (min, max)
            alpha_range: Range for balancing parameter (min, max)
            tau: Time-step of the dual ascent
            DC: True if the first mode is DC(low-frequency) component
            init: Initialization method (0 = all omegas start at 0, 1 = all omegas start uniformly distributed)
            tol: Tolerance for convergence
        """
        self.k_range = k_range
        self.alpha_range = alpha_range
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.best_k = None
        self.best_alpha = None
        
    def _simplified_vmd(self, signal, k, alpha):
        """
        Simplified VMD implementation for when vmdpy is not available.
        Uses empirical mode decomposition as approximation.
        
        Args:
            signal: Input signal
            k: Number of modes
            alpha: Balancing parameter
            
        Returns:
            u: Decomposed modes
            u_hat: Spectra of modes
            omega: Center frequencies
        """
        # Simple frequency-based decomposition
        n = len(signal)
        u = np.zeros((k, n))
        
        # FFT of signal
        signal_fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n)
        
        # Divide frequency spectrum into k bands
        omega = np.linspace(0, 0.5, k + 1)
        
        for i in range(k):
            # Create bandpass filter
            freq_mask = (np.abs(freqs) >= omega[i]) & (np.abs(freqs) < omega[i + 1])
            
            # Apply filter
            mode_fft = signal_fft.copy()
            mode_fft[~freq_mask] = 0
            
            # Inverse FFT
            u[i, :] = np.real(np.fft.ifft(mode_fft))
        
        omega = (omega[:-1] + omega[1:]) / 2
        u_hat = np.fft.fft(u, axis=1)
        
        return u, u_hat, omega
    
    def decompose(self, signal, k=None, alpha=None):
        """
        Decompose signal using VMD.
        
        Args:
            signal: Input time series signal
            k: Number of modes (if None, uses best_k or middle of range)
            alpha: Balancing parameter (if None, uses best_alpha or middle of range)
            
        Returns:
            u: Decomposed modes (k, n_timesteps)
            u_hat: Spectra of modes
            omega: Center frequencies of modes
        """
        if k is None:
            k = self.best_k if self.best_k is not None else (self.k_range[0] + self.k_range[1]) // 2
        if alpha is None:
            alpha = self.best_alpha if self.best_alpha is not None else (self.alpha_range[0] + self.alpha_range[1]) / 2
        
        if VMD_AVAILABLE:
            try:
                u, u_hat, omega = VMD(signal, alpha, self.tau, k, self.DC, self.init, self.tol)
                return u, u_hat, omega
            except Exception as e:
                print(f"VMD failed, using simplified version: {e}")
                return self._simplified_vmd(signal, k, alpha)
        else:
            return self._simplified_vmd(signal, k, alpha)
    
    def compute_decomposition_quality(self, signal, u):
        """
        Compute quality metrics for VMD decomposition.
        
        Args:
            signal: Original signal
            u: Decomposed modes
            
        Returns:
            quality_score: Combined quality score (higher is better)
        """
        # Reconstruction error
        reconstructed = np.sum(u, axis=0)
        reconstruction_error = np.mean((signal - reconstructed) ** 2)
        
        # Orthogonality between modes
        k = u.shape[0]
        orthogonality = 0.0
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                correlation = np.abs(np.corrcoef(u[i], u[j])[0, 1])
                orthogonality += correlation
                count += 1
        
        orthogonality = orthogonality / count if count > 0 else 0.0
        
        # Mode sparsity (prefer well-separated modes)
        mode_variances = np.var(u, axis=1)
        sparsity = np.std(mode_variances) / (np.mean(mode_variances) + 1e-8)
        
        # Combined quality score (lower reconstruction error, lower orthogonality, higher sparsity)
        quality_score = (1.0 / (1.0 + reconstruction_error) - 
                        self.ORTHOGONALITY_WEIGHT * orthogonality + 
                        self.SPARSITY_WEIGHT * sparsity)
        
        return quality_score
    
    def optimize_parameters(self, signal, n_trials=20):
        """
        Optimize VMD parameters using grid search (simplified Bayesian optimization).
        
        Args:
            signal: Input time series signal
            n_trials: Number of optimization trials
            
        Returns:
            best_params: Dictionary with best k and alpha
            best_quality: Best quality score achieved
        """
        best_quality = -np.inf
        best_params = {'k': None, 'alpha': None}
        
        # Grid search over parameter space
        k_values = np.linspace(self.k_range[0], self.k_range[1], 
                              min(n_trials // 4, self.k_range[1] - self.k_range[0] + 1), 
                              dtype=int)
        alpha_values = np.logspace(np.log10(self.alpha_range[0]), 
                                   np.log10(self.alpha_range[1]), 
                                   n_trials // len(k_values))
        
        for k in k_values:
            for alpha in alpha_values:
                try:
                    u, u_hat, omega = self.decompose(signal, k=k, alpha=alpha)
                    quality = self.compute_decomposition_quality(signal, u)
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_params['k'] = k
                        best_params['alpha'] = alpha
                except Exception as e:
                    print(f"Failed for k={k}, alpha={alpha}: {e}")
                    continue
        
        self.best_k = best_params['k']
        self.best_alpha = best_params['alpha']
        
        return best_params, best_quality
    
    def visualize_decomposition(self, signal, u, omega, save_path=None):
        """
        Visualize VMD decomposition quality.
        
        Args:
            signal: Original signal
            u: Decomposed modes
            omega: Center frequencies
            save_path: Optional path to save the figure
        """
        k = u.shape[0]
        n_timesteps = u.shape[1]
        
        fig, axes = plt.subplots(k + 2, 1, figsize=(12, 2 * (k + 2)))
        
        # Original signal
        axes[0].plot(signal, 'k', linewidth=1.5)
        axes[0].set_title('Original Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Each mode
        for i in range(k):
            axes[i + 1].plot(u[i], linewidth=1.5)
            axes[i + 1].set_title(f'Mode {i + 1} (ω = {omega[i]:.4f})')
            axes[i + 1].set_ylabel('Amplitude')
            axes[i + 1].grid(True, alpha=0.3)
        
        # Reconstructed signal
        reconstructed = np.sum(u, axis=0)
        axes[k + 1].plot(signal, 'k', alpha=0.5, label='Original', linewidth=1.5)
        axes[k + 1].plot(reconstructed, 'r--', label='Reconstructed', linewidth=1.5)
        axes[k + 1].set_title('Original vs Reconstructed')
        axes[k + 1].set_xlabel('Time Steps')
        axes[k + 1].set_ylabel('Amplitude')
        axes[k + 1].legend()
        axes[k + 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
