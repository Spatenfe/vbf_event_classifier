"""Variational Autoencoder for minority-class tabular data generation."""
import logging
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ml_framework.core.registry import Registry
from ml_framework.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class _Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout_rate):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class _VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.decoder = _Decoder(latent_dim, hidden_dims, input_dim, dropout_rate)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


@Registry.register_generator("vae")
class VAEGenerator(BaseGenerator):
    """
    Trains a VAE on samples from a target class and generates new synthetic samples.

    Config params:
        hidden_dims     (list[int])  Encoder/decoder hidden layer sizes  [128, 64]
        latent_dim      (int)        Size of the latent space            16
        epochs          (int)        Training epochs                     300
        batch_size      (int)        Mini-batch size                     256
        learning_rate   (float)      Adam learning rate                  1e-3
        kl_weight       (float)      Weight on the KL term (β-VAE)       1.0
        dropout_rate    (float)      Dropout in encoder/decoder          0.1
        log_interval    (int)        Log every N epochs                  50
    """

    def __init__(self, config):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.vae = None
        self.feature_names = None
        self.input_dim = None

    def fit(self, X, feature_names=None):
        self.feature_names = feature_names
        self.input_dim = X.shape[1]

        p = self.params
        hidden_dims  = p.get("hidden_dims", [128, 64])
        latent_dim   = p.get("latent_dim", 16)
        epochs       = p.get("epochs", 300)
        batch_size   = p.get("batch_size", 256)
        lr           = p.get("learning_rate", 1e-3)
        kl_weight    = p.get("kl_weight", 1.0)
        dropout_rate = p.get("dropout_rate", 0.1)
        log_interval = p.get("log_interval", 50)

        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training VAE on %d samples, %d features, device=%s", len(X_scaled), self.input_dim, device)

        self.vae = _VAE(self.input_dim, hidden_dims, latent_dim, dropout_rate).to(device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        loader = DataLoader(
            TensorDataset(torch.tensor(X_scaled)),
            batch_size=batch_size,
            shuffle=True,
        )

        self.vae.train()
        for epoch in range(1, epochs + 1):
            total_recon = 0.0
            total_kl = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x_hat, mu, log_var = self.vae(batch)
                recon = nn.functional.mse_loss(x_hat, batch, reduction="sum")
                kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon + kl_weight * kl
                loss.backward()
                optimizer.step()
                total_recon += recon.item()
                total_kl += kl.item()

            if epoch % log_interval == 0 or epoch == 1:
                n = len(X_scaled)
                logger.info(
                    "VAE Epoch %4d/%d | recon=%.4f | kl=%.4f | total=%.4f",
                    epoch, epochs,
                    total_recon / n,
                    total_kl / n,
                    (total_recon + kl_weight * total_kl) / n,
                )

        self.vae.eval()
        logger.info("VAE training complete.")

    def generate(self, n_samples):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.vae.latent_dim, device=device)
            samples = self.vae.decoder(z).cpu().numpy()
        return self.scaler.inverse_transform(samples)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.vae.state_dict(), os.path.join(output_dir, "vae.pt"))
        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "input_dim": self.input_dim,
                "params": self.params,
            },
            os.path.join(output_dir, "vae_meta.pkl"),
        )
        logger.info("VAE saved to %s", output_dir)

    def load(self, model_dir):
        meta = joblib.load(os.path.join(model_dir, "vae_meta.pkl"))
        self.scaler = meta["scaler"]
        self.feature_names = meta["feature_names"]
        self.input_dim = meta["input_dim"]
        self.params = meta["params"]

        p = self.params
        hidden_dims  = p.get("hidden_dims", [128, 64])
        latent_dim   = p.get("latent_dim", 16)
        dropout_rate = p.get("dropout_rate", 0.1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = _VAE(self.input_dim, hidden_dims, latent_dim, dropout_rate)
        self.vae.load_state_dict(
            torch.load(os.path.join(model_dir, "vae.pt"), map_location=device, weights_only=True)
        )
        self.vae.to(device).eval()
        logger.info("VAE loaded from %s", model_dir)
