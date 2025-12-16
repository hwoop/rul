"""
Particle Filter Implementation for MSDFM

Reference: "Remaining useful life prediction based on a multi-sensor data fusion model"
Li et al., Reliability Engineering and System Safety 208 (2021) 107249

Key Implementation Details:
- Table 1: RUL prediction algorithm using a random sensor group
- Eq. 19: State prediction step
- Eq. 20: Weight update using multivariate normal likelihood
- Eq. 21: RUL PDF (Inverse Gaussian distribution)
- Reference [27]: Fuzzy resampling algorithm
"""

import numpy as np
from scipy.stats import multivariate_normal
from config import Config


class ParticleFilter:
    """
    Particle Filter for joint estimation of degradation rate (η) and system state (x).
    
    Implements the algorithm described in Table 1 of the paper.
    """
    
    def __init__(self, params, sensors, initial_data=None):
        """
        Initialize particle filter.
        
        Args:
            params: MSDFM_Parameters object containing estimated parameters
            sensors: List of sensor names to use
            initial_data: First measurement vector for state initialization (optional)
        """
        self.params = params
        self.sensors = sensors
        self.Ns = Config.NUM_PARTICLES
        
        # ================================================================
        # PARTICLE INITIALIZATION (Section 3.3.1)
        # ================================================================
        # Particles: [η (degradation rate), x (system state)]
        self.particles = np.zeros((self.Ns, 2))
        
        # η is sampled from N(μ_η, σ²_η) - as stated in the paper
        self.particles[:, 0] = np.random.normal(
            params.mu_eta, 
            params.sigma_eta,
            self.Ns
        )
        # Ensure positive degradation rates
        self.particles[:, 0] = np.maximum(self.particles[:, 0], 1e-8)
        
        # x initialized to 0 (Section 3.1: "initial value is 0")
        self.particles[:, 1] = 0.0
        
        # Uniform initial weights: w^i_0 = 1/N_s
        self.weights = np.ones(self.Ns) / self.Ns
        
        # ================================================================
        # OPTIONAL: Initialize from first measurement (for test units)
        # ================================================================
        if initial_data is not None:
            x_init = self._infer_initial_state(initial_data)
            # Add small noise for diversity
            self.particles[:, 1] = x_init + np.random.normal(0, 0.02, self.Ns)
            self.particles[:, 1] = np.clip(self.particles[:, 1], 0, 0.99)
            
            # Perform initial update
            self.update(initial_data)
            self.fuzzy_resampling()

    def _infer_initial_state(self, measurement_vector):
        """
        Infer initial state from first measurement using inverse measurement function.
        
        Uses: x = ψ((y - b) / a, θ) where ψ is inverse of φ
        For φ(x) = x^c: x = ((y - b) / a)^(1/c)
        """
        x_estimates = []
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            y_meas = measurement_vector[i]
            
            try:
                val = (y_meas - p['b']) / p['a']
                if val > 0 and p['c'] > 0:
                    x_est = val ** (1.0 / p['c'])
                    if 0 <= x_est <= 1:
                        x_estimates.append(x_est)
            except (ValueError, ZeroDivisionError):
                continue
        
        if len(x_estimates) == 0:
            return np.zeros(self.Ns)
        
        # Use median for robustness
        return np.full(self.Ns, np.median(x_estimates))
    
    def predict(self):
        """
        State transition step - Eq. 19
        
        x^i_k = x^i_{k-1} + η^i_{k-1} * Δt_k + ω^i_{k-1}
        
        where ω ~ N(0, σ²_B * Δt)
        """
        eta = self.particles[:, 0]
        x_prev = self.particles[:, 1]
        dt = Config.DT
        
        # State transition noise (Brownian motion increment)
        # ω ~ N(0, σ²_B * Δt)
        omega = np.random.normal(0, self.params.sigma_B * np.sqrt(dt), self.Ns)
        
        # State update
        x_new = x_prev + eta * dt + omega
        
        # Physical constraint: x >= 0
        x_new = np.maximum(x_new, 0.0)
        
        self.particles[:, 1] = x_new
    
    def update(self, measurement_vector):
        """
        Weight update using multivariate normal likelihood - Eq. 20
        
        w^i_k ∝ w^i_{k-1} * p(Y_Ω,n,k | x^i_k)
        
        where p(Y|x) = N(Y; A_Ω Φ^i_Ω,k + B_Ω, Σ_Ω)
        """
        P = len(self.sensors)
        x_particles = self.particles[:, 1]
        
        # ================================================================
        # COMPUTE PREDICTED MEASUREMENTS: Y_pred = A * Φ(x) + B
        # ================================================================
        Y_pred = np.zeros((self.Ns, P))
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            
            # Φ(x) = x^c (polynomial function)
            # Clip x to avoid numerical issues
            x_clipped = np.clip(x_particles, 1e-10, 10.0)
            phi_x = x_clipped ** p['c']
            
            # Y = a * Φ(x) + b
            Y_pred[:, i] = p['a'] * phi_x + p['b']
        
        # ================================================================
        # COMPUTE RESIDUALS
        # ================================================================
        Y_obs = np.array(measurement_vector)
        residuals = Y_obs - Y_pred  # (Ns, P)
        
        # ================================================================
        # GET COVARIANCE MATRIX FOR SELECTED SENSORS
        # ================================================================
        try:
            full_sensors = self.params.sensor_list
            indices = [full_sensors.index(s) for s in self.sensors]
            cov = self.params.Cov_matrix[np.ix_(indices, indices)].copy()
        except (AttributeError, ValueError, IndexError):
            cov = self.params.Cov_matrix.copy()
        
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]])
        
        # ================================================================
        # ENSURE NUMERICAL STABILITY OF COVARIANCE
        # ================================================================
        # Add minimal regularization (NOT the large inflation from before)
        epsilon = Config.COVARIANCE_REGULARIZATION
        cov = cov + epsilon * np.eye(P)
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < epsilon:
            cov += (epsilon - eigvals.min() + epsilon) * np.eye(P)
        
        # ================================================================
        # COMPUTE LIKELIHOODS (Eq. 20)
        # ================================================================
        try:
            # Use log-space for numerical stability
            log_likelihoods = multivariate_normal.logpdf(
                residuals,
                mean=np.zeros(P),
                cov=cov,
                allow_singular=True
            )
            
            # Normalize in log-space to prevent underflow
            log_likelihoods = log_likelihoods - log_likelihoods.max()
            likelihoods = np.exp(log_likelihoods)
            
        except Exception as e:
            print(f"  Warning: Likelihood computation failed: {e}")
            likelihoods = np.ones(self.Ns)
        
        # ================================================================
        # UPDATE WEIGHTS
        # ================================================================
        self.weights = self.weights * likelihoods
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-300:
            self.weights = self.weights / weight_sum
        else:
            # Complete weight collapse - reset to uniform
            print("  Warning: Weight collapse detected, resetting to uniform")
            self.weights = np.ones(self.Ns) / self.Ns
    
    def effective_sample_size(self):
        """
        Calculate effective sample size: N_eff = 1 / Σ(w_i²)
        
        Used to determine when resampling is needed.
        """
        return 1.0 / np.sum(self.weights ** 2)
    
    def fuzzy_resampling(self):
        """
        Fuzzy resampling algorithm from Reference [27].
        
        Li et al. (2019) - "A Wiener-process-model-based method for remaining 
        useful life prediction considering unit-to-unit variability"
        
        Process:
        1. Systematic resampling to select particles
        2. Add "fuzzing" noise to duplicated particles to maintain diversity
        """
        Ns = self.Ns
        weights = self.weights.copy()
        
        # ====================================================================
        # CHECK IF RESAMPLING IS NEEDED
        # ====================================================================
        N_eff = self.effective_sample_size()
        if N_eff >= Ns * Config.ESS_THRESHOLD_RATIO:
            return  # No resampling needed
        
        # ====================================================================
        # STEP 1: Systematic Resampling
        # ====================================================================
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Ensure exact sum
        
        # Generate systematic sample points
        u0 = np.random.random() / Ns
        positions = u0 + np.arange(Ns) / Ns
        
        # Find indices
        indexes = np.searchsorted(cumsum, positions)
        indexes = np.clip(indexes, 0, Ns - 1)
        
        # Resample particles
        new_particles = self.particles[indexes].copy()
        
        # ====================================================================
        # STEP 2: Fuzzing (Add noise to maintain diversity)
        # ====================================================================
        # Reference [27]: "noise std = sqrt(variance / Np)"
        # Only fuzz the degradation rate η, not the state x
        
        # Calculate variance of η before resampling
        var_eta = np.var(self.particles[:, 0])
        sigma_fuzz = np.sqrt(var_eta / Ns) if var_eta > 0 else 1e-8
        
        # Find duplicated particles
        unique_idx, counts = np.unique(indexes, return_counts=True)
        
        for idx, count in zip(unique_idx, counts):
            if count > 1:
                # Find all copies of this particle
                mask = (indexes == idx)
                # Add fuzzing noise to η (degradation rate)
                noise = np.random.normal(0, sigma_fuzz, count)
                new_particles[mask, 0] += noise
        
        # Ensure positive η
        new_particles[:, 0] = np.maximum(new_particles[:, 0], 1e-8)
        
        # Update particles and reset weights
        self.particles = new_particles
        self.weights = np.ones(Ns) / Ns
    
    def estimate_rul(self, return_distribution=False):
        """
        RUL prediction using closed-form solution - Eq. 21
        
        The RUL PDF follows an Inverse Gaussian distribution:
        f(l|x̂_k, η̂_k) = (D - x̂_k) / sqrt(2π σ̂²_B l³) * exp(-(l*η̂_k - D + x̂_k)² / (2σ̂²_B l))
        
        The expectation of this distribution is:
        E[L] = (D - x̂_k) / η̂_k
        
        Paper uses MEDIAN for η̂_k and x̂_k (Section 3.3.1)
        """
        # Use median as stated in the paper (Section 3.3.1)
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        
        D = Config.FAILURE_THRESHOLD
        
        # Boundary conditions
        if x_hat >= D:
            return (0.0, 0.0) if return_distribution else 0.0
        
        if eta_hat <= 1e-9:
            # Very slow degradation - return large RUL
            return (1000.0, 500.0) if return_distribution else 1000.0
        
        # Mean RUL (First Passage Time expectation for Wiener process)
        mean_rul = (D - x_hat) / eta_hat
        
        if not return_distribution:
            return max(0, mean_rul)
        
        # ================================================================
        # VARIANCE ESTIMATION (from Inverse Gaussian distribution)
        # ================================================================
        # Var[L] = (D - x̂_k)³ / (η̂_k² * σ²_B)
        # This can be very large for small x, so we also compute particle-based variance
        
        # Method 1: Particle-based variance (more robust)
        rul_particles = []
        for i in range(self.Ns):
            eta_i = self.particles[i, 0]
            x_i = self.particles[i, 1]
            if eta_i > 1e-9 and x_i < D:
                rul_i = (D - x_i) / eta_i
                if rul_i > 0:
                    rul_particles.append(rul_i)
        
        if len(rul_particles) > 10:
            std_rul = np.std(rul_particles)
        else:
            # Fallback: theoretical variance with clipping
            var_theory = (D - x_hat)**3 / (eta_hat**2 * max(self.params.sigma_B**2, 1e-9))
            std_rul = np.sqrt(var_theory)
            # Clip to reasonable range (< 200% CV)
            std_rul = min(std_rul, mean_rul * 2.0)
        
        return max(0, mean_rul), max(0, std_rul)
    
    def get_state_estimate(self):
        """Get current state estimate (median of particles)."""
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        return eta_hat, x_hat
    
    def get_diagnostics(self):
        """Get diagnostic information for debugging."""
        N_eff = self.effective_sample_size()
        return {
            'N_eff': N_eff,
            'N_eff_ratio': N_eff / self.Ns,
            'eta_mean': np.mean(self.particles[:, 0]),
            'eta_std': np.std(self.particles[:, 0]),
            'eta_median': np.median(self.particles[:, 0]),
            'x_mean': np.mean(self.particles[:, 1]),
            'x_std': np.std(self.particles[:, 1]),
            'x_median': np.median(self.particles[:, 1]),
            'weight_max': np.max(self.weights),
            'weight_entropy': -np.sum(self.weights * np.log(self.weights + 1e-300)),
        }
