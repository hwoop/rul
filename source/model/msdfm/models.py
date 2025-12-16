"""
MSDFM Model Parameter Estimation

Reference: "Remaining useful life prediction based on a multi-sensor data fusion model"
Li et al., Reliability Engineering and System Safety 208 (2021) 107249

Key Implementation Details:
- Section 3.1: State transition function (linear Wiener process)
- Section 3.2: Parameter estimation methodology
- Eq. 10-13: State parameters (μ_η, σ²_η, σ²_B)
- Eq. 14-17: Measurement function parameters (a_p, b_p, θ_p)
- Eq. 18: Covariance matrix estimation
- Appendix A & B: Derivation details
"""

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from config import Config

lowess = sm.nonparametric.lowess


class MSDFM_Parameters:
    """
    Container for all MSDFM model parameters.
    
    State Transition Parameters:
        - mu_eta: Mean degradation rate
        - sigma_eta: Std of degradation rate
        - sigma_B: Std of Brownian motion diffusion
        
    Measurement Function Parameters:
        - sensor_params: Dict of {sensor: {'a': a_p, 'b': b_p, 'c': c_p}}
        - Cov_matrix: Covariance matrix of measurement noises
        - sensor_list: List of sensor names
    """
    
    def __init__(self):
        self.mu_eta = None
        self.sigma_eta = None
        self.sigma_B = None
        self.sensor_params = {}
        self.Cov_matrix = None
        self.sensor_list = []

    def estimate_state_params(self, lifetimes):
        """
        Estimate state transition parameters using MLE - Appendix A
        
        The lifetime follows an Inverse Gaussian distribution (Eq. 10):
        f(t) = D / sqrt(2πt²(σ²_η t² + σ²_B t)) × exp(-(μ_η t - D)² / (2(σ²_η t² + σ²_B t)))
        
        Estimates:
            - μ_η: Mean degradation rate (Eq. 12)
            - σ²_η: Variance of degradation rate (Eq. 13)
            - σ²_B: Variance of Brownian motion diffusion
            
        Note on Virtual State:
        =====================
        Section 3.1: "we define a virtual state as a linear Wiener process 
        increasing from 0 to 1"
        
        This means even for systems with nonlinear physical degradation,
        we use a LINEAR virtual state model. The nonlinearity is absorbed
        into the measurement function (see Section 4.2, Figure 5).
        """
        N = len(lifetimes)
        D = Config.FAILURE_THRESHOLD  # = 1.0
        T_k = np.array(lifetimes, dtype=np.float64)
        
        # Validate input
        if N == 0:
            raise ValueError("No lifetime data provided")
        if np.any(T_k <= 0):
            raise ValueError("Lifetimes must be positive")
        
        print(f"[State Params] Estimating from {N} training units...")
        print(f"  Lifetime range: [{T_k.min():.1f}, {T_k.max():.1f}], mean={T_k.mean():.1f}")

        def neg_log_likelihood(sigma_B_tilde):
            """
            Negative log-likelihood function (Eq. 11)
            
            Let σ̃²_B = σ²_B / σ²_η
            
            L(σ̃²_B|T) = -Σ ln(t_{n,K_n}) - (1/2)Σ ln(t²_{n,K_n} + σ̃²_B t_{n,K_n})
                        - (N/2)ln(2π) - (N/2)ln(σ²_η(σ̃²_B)) - N/2
            """
            if sigma_B_tilde <= 0:
                return 1e9
            
            # Calculate μ̂_η (Eq. 12)
            denom_terms = T_k + sigma_B_tilde
            term1 = np.sum(D / denom_terms)
            term2 = np.sum(T_k / denom_terms)
            
            if term2 == 0:
                return 1e9
            
            mu_eta_hat = term1 / term2
            
            # Calculate σ̂²_η (Eq. 13)
            numerator = (mu_eta_hat * T_k - D) ** 2
            denominator = T_k**2 + sigma_B_tilde * T_k
            
            # Avoid division by zero
            denominator = np.maximum(denominator, 1e-10)
            sigma_eta_sq = (1/N) * np.sum(numerator / denominator)
            
            if sigma_eta_sq <= 0:
                return 1e9
            
            # Compute negative log-likelihood
            term_log = np.sum(np.log(denominator))
            nll = 0.5 * term_log + 0.5 * N * np.log(sigma_eta_sq)
            
            return nll

        # Optimize σ̃²_B
        # Reasonable bounds based on typical C-MAPSS data
        result = minimize_scalar(
            neg_log_likelihood, 
            bounds=(1e-3, 500), 
            method='bounded',
            options={'xatol': 1e-8}
        )
        sigma_B_tilde_hat = result.x
        
        # Calculate final estimates
        # μ̂_η (Eq. 12)
        denom_terms = T_k + sigma_B_tilde_hat
        term1 = np.sum(D / denom_terms)
        term2 = np.sum(T_k / denom_terms)
        self.mu_eta = term1 / term2
        
        # σ̂²_η (Eq. 13)
        numerator = (self.mu_eta * T_k - D) ** 2
        denominator = T_k**2 + sigma_B_tilde_hat * T_k
        sigma_eta_sq = (1/N) * np.sum(numerator / denominator)
        self.sigma_eta = np.sqrt(max(sigma_eta_sq, 1e-10))
        
        # σ²_B = σ²_η × σ̃²_B
        self.sigma_B = np.sqrt(max(sigma_eta_sq * sigma_B_tilde_hat, 1e-10))
        
        print(f"[State Params] Estimated:")
        print(f"  μ_η = {self.mu_eta:.6f} (expected ~{D/T_k.mean():.6f} = 1/mean_life)")
        print(f"  σ_η = {self.sigma_eta:.6f}")
        print(f"  σ_B = {self.sigma_B:.6f}")

    def estimate_measurement_params(self, train_df, sensors_to_use):
        """
        Estimate measurement function parameters - Section 3.2
        
        For each sensor p, the measurement function is (Eq. 7):
        y^p_k = a_p × φ(x_k, θ_p) + b_p + v^p_k
        
        We use φ(x) = x^c (polynomial function)
        
        Estimates for each sensor:
            - c_p: Power parameter (Eq. 15)
            - a_p: Scale parameter (Eq. 16)
            - b_p: Location parameter (Eq. 17)
            
        Covariance matrix Σ estimated from residuals (Eq. 18)
        
        IMPORTANT - Handling Nonlinear Degradation:
        ==========================================
        Section 4.2 explains that for nonlinear physical degradation:
        "the nonlinearity of state degradation is transformed into the 
        measurement function. Therefore, parameter c_p in dataset 2 
        acquires higher estimation results than actual values."
        
        This is why c_p may be overestimated for nonlinear systems.
        """
        self.sensor_list = list(sensors_to_use)
        units = train_df['unit_nr'].unique()
        N = len(units)
        
        print(f"\n[Measurement Params] Estimating for {len(sensors_to_use)} sensors...")
        
        # ====================================================================
        # STEP 1: Smooth signals and estimate individual sensor parameters
        # ====================================================================
        smoothed_data = {}  # Store smoothed signals for covariance estimation
        
        for s_idx, sensor in enumerate(sensors_to_use):
            unit_smoothed = []
            
            for unit in units:
                u_data = train_df[train_df['unit_nr'] == unit]
                y = u_data[sensor].values.astype(np.float64)
                t = u_data['time_cycles'].values.astype(np.float64)
                
                # Handle NaNs
                if np.any(np.isnan(y)):
                    y = np.nan_to_num(y, nan=np.nanmean(y))
                
                # LOWESS smoothing [Reference 28: Cleveland & Devlin 1988]
                try:
                    y_smooth = lowess(y, t, frac=Config.SMOOTHING_FRAC, return_sorted=False)
                except Exception:
                    y_smooth = y.copy()
                
                unit_smoothed.append((y_smooth, t, y))  # Store raw y too
            
            smoothed_data[sensor] = unit_smoothed
            
            # ------------------------------------------------------------------
            # Optimize c_p (Eq. 15)
            # ------------------------------------------------------------------
            def objective_c(c):
                """
                Objective function from Eq. 15:
                min_{θ_p} Σ_n [(1/K_n) Σ_k (ψ((ỹ^p_{n,k} - b_p)/a_p, θ_p) - μ_η t_{n,k})²]
                
                For φ(x) = x^c, the inverse is ψ(y) = y^(1/c)
                """
                if c <= 0:
                    return 1e9
                
                # φ(1) - φ(0) for normalization
                phi_1 = 1.0 ** c  # = 1
                phi_0 = 0.0 ** c if c > 0 else 0  # = 0
                denom = phi_1 - phi_0
                
                if abs(denom) < 1e-12:
                    return 1e9
                
                # Calculate â_p and b̂_p for this c (Eq. 16-17)
                sum_a_num = 0.0
                sum_b_num = 0.0
                
                for y_smooth, t, _ in unit_smoothed:
                    y_first = y_smooth[0]
                    y_last = y_smooth[-1]
                    
                    # Eq. 16 numerator: (ỹ_last - ỹ_first)
                    sum_a_num += (y_last - y_first)
                    
                    # Eq. 17 numerator: (ỹ_first × φ(1) - ỹ_last × φ(0))
                    sum_b_num += (y_first * phi_1 - y_last * phi_0)
                
                a_hat = (1/N) * sum_a_num / denom
                b_hat = (1/N) * sum_b_num / denom
                
                # Avoid degenerate cases
                if abs(a_hat) < 1e-12:
                    return 1e9
                
                # Calculate fitting loss (Eq. 15)
                total_loss = 0.0
                
                for y_smooth, t, _ in unit_smoothed:
                    K_n = len(y_smooth)
                    
                    # Inverse function: x = ((y - b) / a)^(1/c)
                    val = (y_smooth - b_hat) / a_hat
                    val = np.clip(val, 1e-12, 1e6)  # Ensure positive
                    
                    try:
                        x_approx = val ** (1.0 / c)
                    except:
                        return 1e9
                    
                    # Loss: (x_approx - μ_η × t)²
                    residuals = x_approx - self.mu_eta * t
                    unit_loss = np.mean(residuals ** 2)
                    total_loss += unit_loss
                
                return total_loss / N
            
            # Optimize c_p within bounds
            c_bounds = Config.C_PARAM_BOUNDS
            result = minimize_scalar(objective_c, bounds=c_bounds, method='bounded')
            c_hat = result.x
            
            # ------------------------------------------------------------------
            # Calculate final a_p and b_p with optimal c_p (Eq. 16-17)
            # ------------------------------------------------------------------
            phi_1 = 1.0 ** c_hat
            phi_0 = 0.0 ** c_hat if c_hat > 0 else 0
            denom = phi_1 - phi_0
            
            sum_a_num = 0.0
            sum_b_num = 0.0
            
            for y_smooth, _, _ in unit_smoothed:
                y_first = y_smooth[0]
                y_last = y_smooth[-1]
                sum_a_num += (y_last - y_first)
                sum_b_num += (y_first * phi_1 - y_last * phi_0)
            
            a_hat = (1/N) * sum_a_num / denom if abs(denom) > 1e-12 else 1.0
            b_hat = (1/N) * sum_b_num / denom if abs(denom) > 1e-12 else 0.0
            
            self.sensor_params[sensor] = {'a': a_hat, 'b': b_hat, 'c': c_hat}
            
            if (s_idx + 1) % 5 == 0 or (s_idx + 1) == len(sensors_to_use):
                print(f"  Progress: {s_idx+1}/{len(sensors_to_use)} sensors")
        
        # ====================================================================
        # STEP 2: Estimate covariance matrix from residuals (Eq. 18)
        # ====================================================================
        print("\n[Covariance Matrix] Estimating from residuals...")
        
        # Collect residuals: v^p_{n,k} = y^p_{n,k} - ỹ^p_{n,k}
        all_residuals = []
        
        for unit_idx, unit in enumerate(units):
            unit_residuals = []
            
            for sensor in sensors_to_use:
                y_smooth, _, y_raw = smoothed_data[sensor][unit_idx]
                residuals = y_raw - y_smooth
                unit_residuals.append(residuals)
            
            # Stack sensors: shape (K_n, P)
            unit_residuals = np.column_stack(unit_residuals)
            all_residuals.append(unit_residuals)
        
        # Concatenate all residuals: shape (sum(K_n), P)
        all_residuals = np.vstack(all_residuals)
        
        # Handle NaNs
        if np.any(np.isnan(all_residuals)):
            nan_count = np.isnan(all_residuals).sum()
            print(f"  Warning: {nan_count} NaNs in residuals, replacing with 0")
            all_residuals = np.nan_to_num(all_residuals, nan=0.0)
        
        # Estimate covariance matrix (Eq. 18)
        self.Cov_matrix = np.cov(all_residuals, rowvar=False)
        
        # Handle single sensor case
        if self.Cov_matrix.ndim == 0:
            self.Cov_matrix = np.array([[float(self.Cov_matrix)]])
        
        # ====================================================================
        # NUMERICAL STABILITY: Regularize covariance matrix
        # ====================================================================
        P = len(sensors_to_use)
        epsilon = Config.COVARIANCE_REGULARIZATION
        
        # Check for NaNs or Infs
        if np.any(np.isnan(self.Cov_matrix)) or np.any(np.isinf(self.Cov_matrix)):
            print("  ERROR: Invalid values in covariance matrix, using diagonal fallback")
            variances = np.var(all_residuals, axis=0)
            self.Cov_matrix = np.diag(np.maximum(variances, epsilon))
            return
        
        # Ensure positive definiteness
        try:
            eigvals = np.linalg.eigvalsh(self.Cov_matrix)
            min_eigval = eigvals.min()
            
            if min_eigval <= epsilon:
                regularization = abs(min_eigval) + epsilon * 2
                self.Cov_matrix += regularization * np.eye(P)
                print(f"  Applied regularization: {regularization:.2e}")
            
            # Verify
            eigvals_after = np.linalg.eigvalsh(self.Cov_matrix)
            print(f"  Covariance eigenvalues: [{eigvals_after.min():.2e}, {eigvals_after.max():.2e}]")
            
        except np.linalg.LinAlgError as e:
            print(f"  ERROR: Eigenvalue check failed: {e}")
            variances = np.var(all_residuals, axis=0)
            self.Cov_matrix = np.diag(np.maximum(variances, epsilon))
        
        print(f"\n[Estimation Complete] {len(sensors_to_use)} sensors parameterized")
        
        # Print summary
        print("\nParameter Summary (first 5 sensors):")
        for i, sensor in enumerate(sensors_to_use[:5]):
            p = self.sensor_params[sensor]
            print(f"  {sensor}: a={p['a']:.4f}, b={p['b']:.4f}, c={p['c']:.4f}")
        if len(sensors_to_use) > 5:
            print(f"  ... and {len(sensors_to_use)-5} more sensors")
