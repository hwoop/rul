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
            z_pred = model.forward_decoder(x_tensor).numpy()
        
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
    
class MonotonicLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MonotonicLinear, self).__init__()
        self.weight_raw = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight_raw)
        nn.init.zeros_(self.bias)
    def forward(self, input):
        return F.linear(input, F.softplus(self.weight_raw), self.bias)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
    def forward(self, h):
        Batch, N, _ = h.size()
        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(self.a(Wh_concat)).squeeze(-1)
        alpha = F.softmax(e, dim=2)
        h_prime = torch.bmm(alpha, Wh)
        return h_prime, alpha
    
class IDSSM(nn.Module):
    def __init__(self, num_sensors, latent_dim=8):
        super(IDSSM, self).__init__()
        self.gat = GATLayer(in_dim=1, out_dim=latent_dim)
        self.pooling = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softmax(dim=1))
        self.mnn = nn.Sequential(
            MonotonicLinear(1, 16),
            nn.Tanh(),
            MonotonicLinear(16, latent_dim)
        )
    def forward_encoder(self, x_sensors):
        h = x_sensors.unsqueeze(-1)
        h_prime, attn_weights = self.gat(h) 
        w = self.pooling(h_prime)
        z = torch.sum(h_prime * w, dim=1)
        return z, attn_weights 
    
    def forward_decoder(self, state_x):
        return self.mnn(state_x)
    
    def forward(self, x_sensors, state_x):
        z_enc, _ = self.forward_encoder(x_sensors) 
        z_dec = self.forward_decoder(state_x)
        return z_enc, z_dec