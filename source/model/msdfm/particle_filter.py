import numpy as np
from scipy.stats import multivariate_normal, invgauss
from config import Config

class ParticleFilter:
    """
    Particle Filter for joint estimation of degradation rate and system state.
    Implements the algorithm in Table 1 with fuzzy resampling [27].
    """
    
    def __init__(self, params, sensors, initial_data=None):
        """
        Initialize particle filter.
        
        Args:
            params: MSDFM_Parameters object
            sensors: List of sensor names to use
            initial_data: IMPORTANT - First measurement for proper initialization
        """
        self.params = params
        self.sensors = sensors
        self.Ns = Config.NUM_PARTICLES
        
        # Particles: [eta, x]
        self.particles = np.zeros((self.Ns, 2))
        
        # ================================================================
        # ADAPTIVE INITIALIZATION
        # ================================================================
        # Wider initial distribution for robustness
        eta_std_multiplier = 3.0  # More conservative
        self.particles[:, 0] = np.random.normal(
            params.mu_eta, 
            params.sigma_eta * eta_std_multiplier,
            self.Ns
        )
        
        # Clamp to reasonable range
        eta_min = params.mu_eta * 0.1
        eta_max = params.mu_eta * 10.0
        self.particles[:, 0] = np.clip(self.particles[:, 0], eta_min, eta_max)
        
        self.particles[:, 1] = 0.0
        self.weights = np.ones(self.Ns) / self.Ns
        
        # Track update count for adaptive inflation
        self.update_count = 0
        
        # ================================================================
        # INITIALIZE WITH FIRST MEASUREMENT
        # ================================================================
        if initial_data is not None:
            x_init_estimates = self._infer_initial_state(initial_data)
            
            self.particles[:, 1] = x_init_estimates + np.random.normal(
                0, 0.1, self.Ns
            )
            self.particles[:, 1] = np.maximum(0, self.particles[:, 1])
            
            # Use high inflation for first update
            self.update(initial_data, inflation_factor=Config.INITIAL_COVARIANCE_INFLATION)
            self.fuzzy_resampling()

    def _infer_initial_state(self, measurement_vector):
        """
        Infer initial state distribution from first measurement.
        Uses inverse measurement function.
        
        Args:
            measurement_vector: First measurement [y_1, ..., y_P]
            
        Returns:
            x_estimates: Array of inferred x values for each particle
        """
        P = len(self.sensors)
        x_estimates = []
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            y_meas = measurement_vector[i]
            
            # Inverse function: x = ((y - b) / a)^(1/c)
            try:
                val = (y_meas - p['b']) / p['a']
                if val > 0:
                    x_est = val ** (1 / p['c'])
                    x_estimates.append(x_est)
            except:
                pass
        
        if len(x_estimates) == 0:
            # Fallback: use small positive values
            return np.random.uniform(0, 0.1, self.Ns)
        
        # Use median of estimates and add particle diversity
        x_median = np.median(x_estimates)
        x_median = np.clip(x_median, 0, 0.5)  # Sanity check
        
        # Generate particles around median
        return np.random.normal(x_median, 0.1, self.Ns)
    
    def predict(self):
        """
        State transition step (Eq 19).
        
        Implements: x_k = x_{k-1} + eta_{k-1} * dt + omega
        where omega ~ N(0, sigma_B^2 * dt)
        """
        eta = self.particles[:, 0]
        x_prev = self.particles[:, 1]
        dt = Config.DT
        
        # State transition noise (Brownian motion increment)
        omega = np.random.normal(0, self.params.sigma_B * np.sqrt(dt), self.Ns)
        
        # Update state
        x_new = x_prev + eta * dt + omega
        
        # Physical constraint: x >= 0 (though rarely violated)
        x_new = np.maximum(x_new, 0.0)
        
        self.particles[:, 1] = x_new
    
    def update(self, measurement_vector, inflation_factor=None):
        """
        Weight update using multivariate likelihood (Eq 20).
        
        Args:
            measurement_vector: Array of sensor measurements [y_1, ..., y_P]
            inflation_factor: If None, uses adaptive strategy
        """
        self.update_count += 1
        
        # ================================================================
        # ADAPTIVE INFLATION STRATEGY
        # ================================================================
        if inflation_factor is None:
            # Decrease inflation as confidence grows
            if self.update_count <= 5:
                inflation_factor = Config.INITIAL_COVARIANCE_INFLATION
            elif self.update_count <= 10:
                inflation_factor = Config.REGULAR_COVARIANCE_INFLATION * 2
            else:
                inflation_factor = Config.REGULAR_COVARIANCE_INFLATION
        
        P = len(self.sensors)
        x_particles = self.particles[:, 1]
        
        # ================================================================
        # PREDICT MEASUREMENTS (with safety bounds)
        # ================================================================
        Y_pred = np.zeros((self.Ns, P))
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            
            # Safe power computation
            x_clipped = np.clip(x_particles, 1e-6, 10.0)
            
            try:
                phi_x = x_clipped ** p['c']
            except:
                phi_x = np.ones_like(x_clipped)
            
            # Prevent extreme predictions
            phi_x = np.clip(phi_x, -1e6, 1e6)
            Y_pred[:, i] = p['a'] * phi_x + p['b']
        
        Y_obs = measurement_vector
        residuals = Y_obs - Y_pred
        
        # ================================================================
        # DIAGNOSTIC: Check residual magnitude
        # ================================================================
        max_residual = np.abs(residuals).max()
        mean_residual = np.abs(residuals).mean()
        
        if max_residual > 10:
            print(f"  [Update {self.update_count}] Large residuals detected:")
            print(f"    Max: {max_residual:.2f}, Mean: {mean_residual:.2f}")
            # Emergency inflation
            inflation_factor = max(inflation_factor, 50.0)
        
        # ================================================================
        # BUILD COVARIANCE MATRIX
        # ================================================================
        try:
            full_sensors = self.params.sensor_list
            current_indices = [full_sensors.index(s) for s in self.sensors]
            cov = self.params.Cov_matrix[np.ix_(current_indices, current_indices)]
        except (AttributeError, ValueError, IndexError):
            cov = self.params.Cov_matrix
        
        if np.ndim(cov) == 0: 
            cov = np.array([[cov]])
        
        # ================================================================
        # ROBUST COVARIANCE INFLATION
        # ================================================================
        # Base inflation
        cov = cov * inflation_factor
        
        # Adaptive regularization based on residuals
        residual_vars = np.var(residuals, axis=0)
        adaptive_reg = np.diag(residual_vars * 0.5)  # 50% of observed variance
        cov = cov + adaptive_reg + np.eye(P) * 1e-4
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-6:
            cov += np.eye(P) * (1e-6 - eigvals.min() + 1e-6)
        
        # ================================================================
        # COMPUTE LIKELIHOODS (log-space for stability)
        # ================================================================
        try:
            log_likelihoods = multivariate_normal.logpdf(
                residuals,
                mean=np.zeros(P),
                cov=cov,
                allow_singular=True
            )
            
            # Shift to prevent underflow
            log_likelihoods = log_likelihoods - log_likelihoods.max()
            likelihoods = np.exp(log_likelihoods)
            
        except Exception as e:
            print(f"  Warning: Likelihood calculation failed: {e}")
            likelihoods = np.ones(self.Ns)
        
        # ================================================================
        # DEFENSIVE MIXTURE
        # ================================================================
        mixture_weight = 0.05  # 5% uniform (increased from 1%)
        likelihoods = (1 - mixture_weight) * likelihoods + \
                    mixture_weight / self.Ns
        
        # Update weights
        self.weights *= likelihoods
        self.weights += 1e-300
        
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            print("  Warning: Complete weight collapse, resetting")
            self.weights = np.ones(self.Ns) / self.Ns
            return
        
        # ================================================================
        # MONITOR AND RECOVER
        # ================================================================
        N_eff = self.effective_sample_size()
        
        if N_eff < self.Ns * Config.MIN_EFFECTIVE_SAMPLE_SIZE_RATIO:
            print(f"  [Update {self.update_count}] Critical N_eff={N_eff:.1f}, recovering...")
            
            # Emergency diversity injection
            eta_std = max(np.std(self.particles[:, 0]), self.params.sigma_eta * 0.5)
            x_std = max(np.std(self.particles[:, 1]), 0.05)
            
            self.particles[:, 0] += np.random.normal(0, eta_std * 0.3, self.Ns)
            self.particles[:, 1] += np.random.normal(0, x_std * 0.3, self.Ns)
            self.particles[:, 1] = np.maximum(0, self.particles[:, 1])
            
            # Partial weight reset
            self.weights = 0.5 * self.weights + 0.5 / self.Ns
    
    def effective_sample_size(self):
        """
        Calculate effective sample size: N_eff = 1 / sum(w_i^2)
        
        This metric indicates particle degeneracy:
        - N_eff ≈ Ns: Good diversity
        - N_eff << Ns: Severe degeneracy (few particles have most weight)
        """
        return 1.0 / np.sum(self.weights ** 2)
    
    def fuzzy_resampling(self):
        """
        Fuzzy resampling algorithm from Reference [27].
        
        Process (Section 3.3.1, Table 1):
        1. Systematic resampling to select particles
        2. Add "fuzzing" noise to duplicated particles to maintain diversity
        
        The fuzzing noise prevents particle collapse and improves long-term
        tracking performance.
        """
        Ns = self.Ns
        weights = self.weights
        
        # ====================================================================
        # Step 1: Systematic Resampling
        # ====================================================================
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Ensure exact sum
        
        # Systematic sampling
        u = np.random.random()
        positions = (u + np.arange(Ns)) / Ns
        indexes = np.searchsorted(cumulative_sum, positions)
        
        # Resample particles
        new_particles = self.particles[indexes].copy()
        
        # ====================================================================
        # Step 2: Fuzzing (Add noise to duplicated particles)
        # ====================================================================
        # Identify which particles were selected multiple times
        unique, counts = np.unique(indexes, return_counts=True)
        
        # Calculate fuzzing noise std based on particle variance
        # Following [27]: "noise std = sqrt(variance / Np)"
        var_eta = np.var(self.particles[:, 0])
        
        # Fuzzing noise standard deviation
        # Note: Paper uses variance BEFORE resampling
        sigma_noise = np.sqrt(var_eta / Ns) if var_eta > 0 else 1e-6
        
        # Apply fuzzing to duplicated particles
        for idx, count in zip(unique, counts):
            if count > 1:  # Particle was selected multiple times
                # Find all copies
                mask = (indexes == idx)
                
                # Add Gaussian noise to eta (degradation rate parameter)
                # Only the RATE is fuzzed, not the state x
                noise = np.random.normal(0, sigma_noise, count)
                new_particles[mask, 0] += noise
        
        # Update particles
        self.particles = new_particles
        
        # Reset weights to uniform (standard after resampling)
        self.weights = np.ones(Ns) / Ns
    
    def estimate_rul(self, return_distribution=False):
        """
        RUL prediction following Eq 21 with numerical stability.
        
        Args:
            return_distribution: If True, returns (mean, std) of RUL distribution
                            If False, returns only mean RUL
        
        Returns:
            If return_distribution=False: 
                rul_mean (scalar)
            If return_distribution=True:
                (rul_mean, rul_std) tuple
        """
        # Use median for robustness (stated in Section 3.3.1)
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        
        D = Config.FAILURE_THRESHOLD
        
        # ================================================================
        # BOUNDARY CONDITIONS
        # ================================================================
        if x_hat >= D:
            return (0.0, 0.0) if return_distribution else 0.0
        
        if eta_hat <= 1e-9:
            return (1000.0, 500.0) if return_distribution else 1000.0
        
        # ================================================================
        # MEAN RUL (First Passage Time)
        # ================================================================
        mean_rul = (D - x_hat) / eta_hat
        
        if not return_distribution:
            return mean_rul
        
        # ================================================================
        # ROBUST VARIANCE ESTIMATION
        # ================================================================
        # Problem: Theoretical variance from Eq 21 explodes for small x
        # Solution: Use empirical variance from particles + theoretical correction
        
        # Method 1: Particle-based variance (most robust)
        rul_particles = []
        for i in range(min(1000, self.Ns)):  # Sample for efficiency
            eta_i = self.particles[i, 0]
            x_i = self.particles[i, 1]
            
            if eta_i > 1e-9 and x_i < D:
                rul_i = (D - x_i) / eta_i
                rul_particles.append(rul_i)
        
        if len(rul_particles) > 10:
            # Use empirical variance
            particle_var = np.var(rul_particles)
            particle_std = np.sqrt(particle_var)
            
            # Sanity check: std should be reasonable relative to mean
            if particle_std < mean_rul * 2:
                return mean_rul, particle_std
        
        # Method 2: Theoretical variance with clipping (Eq 21 with safety)
        numerator = (D - x_hat) ** 3
        denominator = (eta_hat ** 2) * (self.params.sigma_B ** 2)
        
        if denominator > 0:
            var_rul_theory = numerator / denominator
            
            # ================================================================
            # CRITICAL: Clip theoretical variance to reasonable range
            # ================================================================
            # Based on uncertainty propagation:
            # - Early stage (x small): High uncertainty
            # - Late stage (x near D): Low uncertainty
            
            # Adaptive clipping based on degradation progress
            progress = x_hat / D  # 0 to 1
            
            if progress < 0.1:
                # Very early: cap at 200% CV
                max_std = mean_rul * 2.0
            elif progress < 0.5:
                # Early-mid: cap at 100% CV
                max_std = mean_rul * 1.0
            else:
                # Late stage: cap at 50% CV
                max_std = mean_rul * 0.5
            
            std_rul_theory = np.sqrt(var_rul_theory)
            std_rul = min(std_rul_theory, max_std)
            
            return mean_rul, std_rul
        
        # Method 3: Fallback - use coefficient of variation from particles
        eta_cv = np.std(self.particles[:, 0]) / (np.mean(self.particles[:, 0]) + 1e-9)
        x_cv = np.std(self.particles[:, 1]) / (np.mean(self.particles[:, 1]) + 1e-9)
        
        # Combined uncertainty
        combined_cv = np.sqrt(eta_cv**2 + x_cv**2)
        std_rul = mean_rul * min(combined_cv, 1.0)  # Cap at 100% CV
        
        return mean_rul, std_rul
    
    def get_state_estimate(self):
        """
        Get current state estimate (median of particles).
        
        Returns:
            (eta_hat, x_hat): Estimated degradation rate and state
        """
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        return eta_hat, x_hat
    

    def get_diagnostics(self):
        """Get detailed diagnostic information."""
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
            'weight_min': np.min(self.weights),
            'weight_entropy': -np.sum(self.weights * np.log(self.weights + 1e-300)),
            'update_count': self.update_count
        }            