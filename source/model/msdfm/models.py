
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from config import Config

lowess = sm.nonparametric.lowess

class MSDFM_Parameters:
    def __init__(self):
        self.mu_eta = None
        self.sigma_eta = None
        self.sigma_B = None
        self.sensor_params = {} 
        self.Cov_matrix = None 
        self.sensor_list = []

    def estimate_state_params(self, lifetimes):
        """
        Estimate state transition parameters using MLE (Appendix A).
        
        Estimates:
            - mu_eta: Mean degradation rate (Eq 12)
            - sigma_eta: Std of degradation rate (Eq 13)
            - sigma_B: Std of Brownian motion diffusion term
        
        NOTE: The virtual state is always assumed to follow LINEAR degradation (Eq 6).
        Even for systems with nonlinear physical degradation, we use this linear
        virtual state model. The nonlinearity is absorbed into the measurement
        function parameters (see estimate_measurement_params).
        """
        N = len(lifetimes)
        D = Config.FAILURE_THRESHOLD
        T_k = lifetimes 

        def neg_log_likelihood(sigma_B_tilde):
            """Negative log-likelihood from Eq 11"""
            if sigma_B_tilde <= 0: 
                return 1e9
            
            term1 = np.sum(D / (T_k + sigma_B_tilde))
            term2 = np.sum(T_k / (T_k + sigma_B_tilde))
            mu_eta_hat = term1 / term2 if term2 != 0 else 0
            
            numerator = (mu_eta_hat * T_k - D)**2
            denominator = T_k**2 + sigma_B_tilde * T_k
            sigma_eta_sq = (1/N) * np.sum(numerator / denominator)
            
            # Prevent log(0)
            if sigma_eta_sq <= 0:
                return 1e9
            
            term_log = np.sum(np.log(T_k**2 + sigma_B_tilde * T_k))
            nll = 0.5 * term_log + 0.5 * N * np.log(sigma_eta_sq)
            return nll

        # Optimize
        res = minimize_scalar(neg_log_likelihood, bounds=(1e-6, 100), method='bounded')
        sigma_B_tilde_hat = res.x
        
        # Calculate final estimates (Eq 12-13)
        term1 = np.sum(D / (T_k + sigma_B_tilde_hat))
        term2 = np.sum(T_k / (T_k + sigma_B_tilde_hat))
        self.mu_eta = term1 / term2
        
        numerator = (self.mu_eta * T_k - D)**2
        denominator = T_k**2 + sigma_B_tilde_hat * T_k
        sigma_eta_sq = (1/N) * np.sum(numerator / denominator)
        self.sigma_eta = np.sqrt(max(0, sigma_eta_sq))  # Safety
        self.sigma_B = np.sqrt(max(0, sigma_eta_sq * sigma_B_tilde_hat))
        
        print(f"[State Params] mu_eta={self.mu_eta:.6f}, "
              f"sigma_eta={self.sigma_eta:.6f}, sigma_B={self.sigma_B:.6f}")

    def estimate_measurement_params(self, train_df, sensors_to_use):
        """
        Estimate measurement function parameters (Section 3.2).
        
        For each sensor p, estimates:
            - a_p: Scale parameter (Eq 16)
            - b_p: Location parameter (Eq 17)  
            - c_p: Exponent in φ(x) = x^c (Eq 15)
        
        CRITICAL NOTE on Nonlinear Physical Degradation:
        ================================================
        The virtual state x(t) is ALWAYS modeled as a linear Wiener process (Eq 6).
        
        For systems with NONLINEAR physical degradation (e.g., α*β*t^(β-1) as in Eq 24):
        
        1. The nonlinearity is TRANSFORMED into the measurement function parameters
        2. This causes c_p to be estimated HIGHER than the true values
        3. This is INTENTIONAL and allows the method to handle various degradation types
        
        As stated in Section 4.2:
        "For nonlinear state degradation cases, the nonlinearity of state degradation 
        is transformed into the measurement function. Therefore, parameter c_p in 
        dataset 2 acquires higher estimation results than actual values."
        
        This is why Figure 5 shows c_p overestimation for nonlinear processes.
        ================================================
        """
        self.sensor_list = sensors_to_use
        
        smoothed_data = {}
        units = train_df['unit_nr'].unique()
        N = len(units)
        
        # ========================================================================
        # Step 1: Individual Sensor Parameter Estimation
        # ========================================================================
        print(f"[Measurement Params] Estimating for {len(sensors_to_use)} sensors...")
        
        for s_idx, sensor in enumerate(sensors_to_use):
            # Smooth signals using LOWESS (Local regression)
            unit_smoothed = []
            for unit in units:
                u_data = train_df[train_df['unit_nr'] == unit]
                y = u_data[sensor].values
                
                # Handle NaNs
                if np.isnan(y).any():
                    print(f"  Warning: NaNs in {sensor} unit {unit}, replacing with 0")
                    y = np.nan_to_num(y, nan=0.0)
                    
                t = u_data['time_cycles'].values
                
                # LOWESS smoothing
                try:
                    y_smooth = lowess(y, t, frac=Config.SMOOTHING_FRAC, return_sorted=False)
                except Exception as e:
                    print(f"  Warning: LOWESS failed for {sensor} unit {unit}: {e}")
                    y_smooth = y  # Fallback to raw
                    
                unit_smoothed.append((y_smooth, t))
            
            smoothed_data[sensor] = unit_smoothed

            # Optimize c_p using Eq 15
            def objective_c(c):
                if c <= 0: 
                    return 1e9
                
                loss = 0
                sum_num_a = 0
                sum_num_b = 0
                denom = 1**c - 0**c  # φ(1) - φ(0) for x^c
                
                if abs(denom) < 1e-9: 
                    return 1e9

                # Calculate a_hat and b_hat for this c (Eq 16-17)
                for y_smooth, t in unit_smoothed:
                    y_last = y_smooth[-1]
                    y_first = y_smooth[0]
                    sum_num_a += (y_last - y_first)
                    sum_num_b += (y_first * 1**c - y_last * 0**c)
                
                a_hat = (1/N) * sum_num_a / denom
                b_hat = (1/N) * sum_num_b / denom
                
                if abs(a_hat) < 1e-6: 
                    return 1e9 
                
                # Calculate loss (Eq 15)
                for y_smooth, t in unit_smoothed:
                    # Inverse function: x = ((y - b)/a)^(1/c)
                    val = (y_smooth - b_hat) / a_hat
                    val = np.clip(val, 1e-9, 1e6)  # Prevent invalid values
                    x_approx = val**(1/c)
                    
                    # Loss: (x_approx - mu_eta * t)^2
                    term = (x_approx - self.mu_eta * t)**2
                    loss += np.mean(term)
                    
                return loss

            # Optimize c_p
            res = minimize_scalar(objective_c, bounds=(0.1, 5.0), method='bounded')
            c_hat = res.x
            
            # Calculate final a_p and b_p
            sum_num_a = 0
            sum_num_b = 0
            for y_smooth, _ in unit_smoothed:
                sum_num_a += (y_smooth[-1] - y_smooth[0])
                sum_num_b += y_smooth[0] 
            
            denom = 1**c_hat - 0**c_hat
            a_hat = (1/N) * sum_num_a / denom if abs(denom) > 1e-9 else 1.0
            b_hat = (1/N) * sum_num_b
            
            self.sensor_params[sensor] = {'a': a_hat, 'b': b_hat, 'c': c_hat}
            
            if (s_idx + 1) % 5 == 0 or (s_idx + 1) == len(sensors_to_use):
                print(f"  Progress: {s_idx+1}/{len(sensors_to_use)} sensors completed")

        # ========================================================================
        # Step 2: Covariance Matrix Estimation (Section 3.2, Eq 18)
        # ========================================================================
        print("[Covariance Matrix] Estimating from residuals...")
        
        residuals_list = []
        for unit_idx, unit in enumerate(units):
            u_data = train_df[train_df['unit_nr'] == unit]
            K_n = len(u_data)
            unit_resids = np.zeros((K_n, len(sensors_to_use)))
            
            for i, sensor in enumerate(sensors_to_use):
                y_raw = u_data[sensor].values
                y_smooth = smoothed_data[sensor][unit_idx][0]
                unit_resids[:, i] = y_raw - y_smooth
            
            residuals_list.append(unit_resids)
            
        all_residuals = np.vstack(residuals_list)
        
        # Handle NaNs in residuals
        if np.isnan(all_residuals).any():
            nan_count = np.isnan(all_residuals).sum()
            print(f"  Warning: {nan_count} NaNs in residuals, replacing with 0")
            all_residuals = np.nan_to_num(all_residuals, nan=0.0)
        
        # Calculate covariance
        self.Cov_matrix = np.cov(all_residuals, rowvar=False)
        
        # Ensure proper shape
        if self.Cov_matrix.ndim == 0:
            self.Cov_matrix = self.Cov_matrix.reshape(1, 1)
        
        # ========================================================================
        # NUMERICAL STABILITY: Regularize Covariance Matrix
        # ========================================================================
        P = len(sensors_to_use)
        epsilon_base = 1e-4  # Base regularization
        
        # Check for NaNs
        if np.isnan(self.Cov_matrix).any():
            print("  ERROR: NaNs in Covariance Matrix, using Identity!")
            self.Cov_matrix = np.eye(P) * 0.01
            return
        
        # Check positive definiteness
        try:
            eigvals = np.linalg.eigvalsh(self.Cov_matrix)
            min_eigval = eigvals.min()
            
            if min_eigval <= 0:
                print(f"  Warning: Covariance not positive definite (min eigval: {min_eigval:.2e})")
                # Add sufficient regularization
                regularization = abs(min_eigval) + epsilon_base
                self.Cov_matrix += regularization * np.eye(P)
                print(f"  Applied regularization: {regularization:.2e}")
            else:
                # Still add small jitter for numerical stability
                self.Cov_matrix += epsilon_base * np.eye(P)
                
            # Verify after regularization
            eigvals_after = np.linalg.eigvalsh(self.Cov_matrix)
            print(f"  Covariance eigenvalues: [{eigvals_after.min():.2e}, {eigvals_after.max():.2e}]")
            
        except np.linalg.LinAlgError as e:
            print(f"  ERROR: Eigenvalue computation failed: {e}")
            print("  Using diagonal covariance as fallback")
            variances = np.diag(self.Cov_matrix)
            self.Cov_matrix = np.diag(np.maximum(variances, epsilon_base))
        
        print(f"[Estimation Complete] {len(sensors_to_use)} sensors parameterized\n")