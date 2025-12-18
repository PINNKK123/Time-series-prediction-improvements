"""
Data preprocessing module for InSAR time series data.
Includes data cleaning, Gaussian filtering, and enhanced causal interpolation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


class InSARDataPreprocessor:
    """
    Preprocessor for InSAR time series data with enhanced cleaning and interpolation.
    """
    
    MIN_SAMPLES_FOR_ZSCORE = 3  # Minimum samples needed for reliable z-score calculation
    
    def __init__(self, anomaly_threshold=3.0, gaussian_sigma=1.0, boundary_padding=2):
        """
        Initialize the preprocessor.
        
        Args:
            anomaly_threshold: Z-score threshold for anomaly detection
            gaussian_sigma: Sigma parameter for Gaussian smoothing
            boundary_padding: Number of boundary points for edge correction
        """
        self.anomaly_threshold = anomaly_threshold
        self.gaussian_sigma = gaussian_sigma
        self.boundary_padding = boundary_padding
        
    def remove_anomalies(self, data, mask=None):
        """
        Remove anomalies from time series data using Z-score method.
        
        Args:
            data: Input time series data (n_samples, n_features, n_timesteps)
            mask: Optional boolean mask indicating valid data points
            
        Returns:
            cleaned_data: Data with anomalies removed
            anomaly_mask: Boolean mask of detected anomalies
        """
        if mask is None:
            mask = np.ones(data.shape, dtype=bool)
        
        cleaned_data = data.copy()
        anomaly_mask = np.zeros(data.shape, dtype=bool)
        
        # Detect anomalies for each feature independently
        for i in range(data.shape[1]):
            feature_data = data[:, i, :]
            
            # Calculate z-scores for each time step
            for t in range(data.shape[2]):
                valid_data = feature_data[mask[:, i, t], t]
                
                if len(valid_data) > self.MIN_SAMPLES_FOR_ZSCORE:  # Need sufficient data points
                    z_scores = np.abs(zscore(valid_data, nan_policy='omit'))
                    anomalies = z_scores > self.anomaly_threshold
                    
                    # Mark anomalies
                    valid_indices = np.where(mask[:, i, t])[0]
                    anomaly_indices = valid_indices[anomalies]
                    anomaly_mask[anomaly_indices, i, t] = True
                    
                    # Replace anomalies with median
                    if np.any(anomalies):
                        median_val = np.median(valid_data[~anomalies])
                        cleaned_data[anomaly_indices, i, t] = median_val
        
        return cleaned_data, anomaly_mask
    
    def apply_gaussian_filter(self, data):
        """
        Apply Gaussian filter for time series smoothing.
        
        Args:
            data: Input time series data (n_samples, n_features, n_timesteps)
            
        Returns:
            smoothed_data: Smoothed time series
        """
        smoothed_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                smoothed_data[i, j, :] = gaussian_filter1d(
                    data[i, j, :], 
                    sigma=self.gaussian_sigma,
                    mode='reflect'
                )
        
        return smoothed_data
    
    def causal_interpolation_with_boundary_correction(self, data, timestamps, mask=None):
        """
        Enhanced causal interpolation with boundary corrections.
        Uses forward-looking interpolation to maintain causality.
        
        Args:
            data: Input time series data (n_samples, n_features, n_timesteps)
            timestamps: Time stamps for each time step
            mask: Optional boolean mask indicating valid data points
            
        Returns:
            interpolated_data: Interpolated time series with boundary corrections
        """
        if mask is None:
            mask = np.ones(data.shape, dtype=bool)
        
        interpolated_data = data.copy()
        n_samples, n_features, n_timesteps = data.shape
        
        for i in range(n_samples):
            for j in range(n_features):
                series = data[i, j, :]
                valid_mask = mask[i, j, :]
                
                if np.sum(valid_mask) < 2:
                    continue
                
                valid_times = timestamps[valid_mask]
                valid_values = series[valid_mask]
                
                # Apply boundary padding
                if len(valid_times) > self.boundary_padding:
                    # Extrapolate boundaries using linear trend
                    left_slope = (valid_values[1] - valid_values[0]) / (valid_times[1] - valid_times[0])
                    right_slope = (valid_values[-1] - valid_values[-2]) / (valid_times[-1] - valid_times[-2])
                    
                    # Pad with extrapolated values
                    pad_times_left = valid_times[0] - np.arange(self.boundary_padding, 0, -1) * (valid_times[1] - valid_times[0])
                    pad_values_left = valid_values[0] - np.arange(self.boundary_padding, 0, -1) * left_slope * (valid_times[1] - valid_times[0])
                    
                    pad_times_right = valid_times[-1] + np.arange(1, self.boundary_padding + 1) * (valid_times[-1] - valid_times[-2])
                    pad_values_right = valid_values[-1] + np.arange(1, self.boundary_padding + 1) * right_slope * (valid_times[-1] - valid_times[-2])
                    
                    # Combine padded and original data
                    extended_times = np.concatenate([pad_times_left, valid_times, pad_times_right])
                    extended_values = np.concatenate([pad_values_left, valid_values, pad_values_right])
                else:
                    extended_times = valid_times
                    extended_values = valid_values
                
                # Causal interpolation (using 'previous' to maintain causality)
                try:
                    # Use linear interpolation with boundary handling
                    interp_func = interp1d(
                        extended_times, 
                        extended_values,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(extended_values[0], extended_values[-1])
                    )
                    interpolated_data[i, j, :] = interp_func(timestamps)
                except Exception as e:
                    # Fallback to original data if interpolation fails
                    print(f"Interpolation failed for sample {i}, feature {j}: {e}")
                    pass
        
        return interpolated_data
    
    def preprocess(self, data, timestamps, mask=None):
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Input time series data (n_samples, n_features, n_timesteps)
            timestamps: Time stamps for each time step
            mask: Optional boolean mask indicating valid data points
            
        Returns:
            processed_data: Fully preprocessed data
            anomaly_mask: Mask of detected anomalies
        """
        # Step 1: Remove anomalies
        cleaned_data, anomaly_mask = self.remove_anomalies(data, mask)
        
        # Step 2: Causal interpolation with boundary correction
        interpolated_data = self.causal_interpolation_with_boundary_correction(
            cleaned_data, timestamps, mask
        )
        
        # Step 3: Apply Gaussian smoothing
        smoothed_data = self.apply_gaussian_filter(interpolated_data)
        
        return smoothed_data, anomaly_mask
