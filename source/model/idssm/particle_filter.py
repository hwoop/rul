import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleFilterRUL:
    def __init__(self, num_particles, drift_mean, drift_std, measurement_noise_std):
        self.N = num_particles
        self.drift_mean = drift_mean
        self.drift_std = drift_std
        self.R_std = measurement_noise_std
        self.particles = np.zeros((self.N, 2))
        self.weights = np.ones(self.N) / self.N
        self.rng = np.random.default_rng(2024)

    def initialize(self):
        self.particles[:, 0] = np.random.uniform(0, 0.01, self.N)
        self.particles[:, 1] = np.random.normal(self.drift_mean, self.drift_std, self.N)
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 1e-5) 

    def predict(self):
        process_noise_x = np.random.normal(0, 1e-4, self.N) 
        process_noise_eta = np.random.normal(0, 1e-5, self.N) 
        self.particles[:, 0] += self.particles[:, 1] + process_noise_x
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 1.2)   
        self.particles[:, 1] += process_noise_eta
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 1e-6)
   
    def update(self, z_obs, model):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(self.particles[:, 0]).unsqueeze(1)
            z_pred = model.decode(x_tensor).numpy()
        
        diff = z_pred - z_obs[None, :]
        dist2 = np.sum(diff**2, axis=1)

        sigma2 = self.R_std**2
        log_likelihood = -0.5 * dist2 / sigma2

        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()             
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        self.weights = weights

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def resample(self):
        indices = self.systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.N)

    def systematic_resample(self, weights):
        N = len(weights)
        positions = (self.rng.random() + np.arange(N)) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def estimate_rul(self, current_time):
        x = np.clip(self.particles[:, 0], 0.0, 1.2)
        eta = np.maximum(self.particles[:, 1], 1e-6)
        rem_life = (1.0 - x) / eta
        rem_life = np.maximum(rem_life, 0)
        pred_rul = np.median(rem_life)        
        return pred_rul