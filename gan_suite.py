"""
GAN Architectures & Evaluation Suite
Based on: "Generative Adversarial Networks for Synthetic Data Generation:
A Comprehensive Survey of Architectures, Applications and Challenges"
Covers: Vanilla GAN, DCGAN, WGAN, WGAN-GP, SN-GAN, BigGAN (conditional),
        StyleGAN2-lite, CTGAN (tabular), TimeGAN (time-series)
Evaluation: FID, Inception Score, TSTR, Discriminative/Predictive scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. DATA LOADERS
# ─────────────────────────────────────────────

def get_mnist_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                         download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_cifar10_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def make_tabular_dataset(n=2000, n_features=10):
    """Synthetic tabular dataset (mixed continuous + discrete)."""
    X = np.random.randn(n, n_features).astype(np.float32)
    X[:, 0] = (X[:, 0] > 0).astype(np.float32)   # binary column
    y = (X[:, 1] + X[:, 2] > 0).astype(np.float32)
    return X, y


def make_timeseries_dataset(n=500, seq_len=24, n_features=5):
    """Synthetic stock-like time series."""
    data = []
    for _ in range(n):
        series = np.cumsum(np.random.randn(seq_len, n_features) * 0.01, axis=0).astype(np.float32)
        data.append(series)
    return np.array(data)   # (n, seq_len, n_features)


# ─────────────────────────────────────────────
# 2. FOUNDATIONAL FRAMEWORKS
# ─────────────────────────────────────────────

class VanillaGenerator(nn.Module):
    """Original Goodfellow et al. 2014 MLP generator."""
    def __init__(self, latent_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),        nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),    nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)


class VanillaDiscriminator(nn.Module):
    """Original MLP discriminator."""
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),  nn.LeakyReLU(0.2),
            nn.Linear(512, 256),      nn.LeakyReLU(0.2),
            nn.Linear(256, 1),        nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class DCGANGenerator(nn.Module):
    """Radford et al. 2016 – fully convolutional with batch norm."""
    def __init__(self, latent_dim=100, channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # 1x1 → 4x4
            nn.ConvTranspose2d(latent_dim, features*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features*8), nn.ReLU(True),
            # 4x4 → 8x8
            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4), nn.ReLU(True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2), nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(features*2, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class DCGANDiscriminator(nn.Module):
    """DCGAN discriminator with batch norm."""
    def __init__(self, channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).view(-1, 1)


# ─────────────────────────────────────────────
# 3. STABILITY-ORIENTED MODELS
# ─────────────────────────────────────────────

class WGANDiscriminator(nn.Module):
    """Arjovsky et al. 2017 – Wasserstein critic (no sigmoid)."""
    def __init__(self, channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, 1, 4, 1, 0, bias=False)   # no activation
        )
    def forward(self, x):
        return self.net(x).view(-1)

    def clip_weights(self, clip_val=0.01):
        for p in self.parameters():
            p.data.clamp_(-clip_val, clip_val)


def gradient_penalty(critic, real, fake, device=DEVICE):
    """WGAN-GP gradient penalty (Gulrajani et al. 2017), λ=10."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((grad.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()
    return gp


class SNDiscriminator(nn.Module):
    """Miyato et al. 2018 – Spectral Normalisation GAN (SN-GAN)."""
    def __init__(self, channels=1, features=64):
        super().__init__()
        SN = nn.utils.spectral_norm
        self.net = nn.Sequential(
            SN(nn.Conv2d(channels, features, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(features, features*2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(features*4, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).view(-1, 1)


# ─────────────────────────────────────────────
# 4. DOMAIN-SPECIFIC: CONDITIONAL (BigGAN-style)
# ─────────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    """
    Conditional GAN generator (Mirza & Osindero 2014 / BigGAN-inspired).
    Embeds class label and concatenates with noise.
    """
    def __init__(self, latent_dim=100, n_classes=10, channels=1, features=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, n_classes)
        in_dim = latent_dim + n_classes
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, features*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features*8), nn.ReLU(True),
            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4), nn.ReLU(True),
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2), nn.ReLU(True),
            nn.ConvTranspose2d(features*2, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z, labels):
        c = self.embed(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x.view(x.size(0), -1, 1, 1))


class ConditionalDiscriminator(nn.Module):
    """Conditional discriminator – concatenates one-hot label to input."""
    def __init__(self, channels=1, n_classes=10, features=64, img_size=32):
        super().__init__()
        self.embed = nn.Embedding(n_classes, img_size * img_size)
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, labels):
        c = self.embed(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([x, c], dim=1)
        return self.net(x).view(-1, 1)


# ─────────────────────────────────────────────
# 5. DOMAIN-SPECIFIC: CTGAN (Tabular)
# ─────────────────────────────────────────────

class CTGANGenerator(nn.Module):
    """
    Xu et al. NeurIPS 2019 – CTGAN for tabular data.
    Uses mode-specific normalisation (simplified: BN per layer).
    """
    def __init__(self, latent_dim=128, data_dim=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, data_dim),   nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)


class CTGANDiscriminator(nn.Module):
    def __init__(self, data_dim=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),   nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),        nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 6. DOMAIN-SPECIFIC: TimeGAN (Sequential)
# ─────────────────────────────────────────────

class TimeGANEmbedder(nn.Module):
    """Yoon et al. NeurIPS 2019 – GRU-based temporal embedder."""
    def __init__(self, input_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.fc(h))


class TimeGANRecovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, output_dim)
    def forward(self, h):
        out, _ = self.gru(h)
        return self.fc(out)


class TimeGANGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, n_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, z):
        h, _ = self.gru(z)
        return torch.sigmoid(self.fc(h))


class TimeGANDiscriminator(nn.Module):
    def __init__(self, hidden_dim, n_layers=3):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        out, _ = self.gru(h)
        return torch.sigmoid(self.fc(out[:, -1, :]))


# ─────────────────────────────────────────────
# 7. TRAINING LOOPS
# ─────────────────────────────────────────────

def train_vanilla_gan(loader, latent_dim=100, epochs=5, device=DEVICE):
    """Trains Vanilla GAN with BCE loss (non-saturating variant)."""
    img_dim = 32 * 32
    G = VanillaGenerator(latent_dim, img_dim).to(device)
    D = VanillaDiscriminator(img_dim).to(device)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    losses = {"G": [], "D": []}

    for epoch in range(epochs):
        g_loss_ep, d_loss_ep = 0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # Discriminator step
            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).detach()
            loss_D = criterion(D(imgs), real_lbl) + criterion(D(fake), fake_lbl)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator step
            z = torch.randn(B, latent_dim, device=device)
            loss_G = criterion(D(G(z)), real_lbl)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_loss_ep += loss_G.item(); d_loss_ep += loss_D.item()

        n = len(loader)
        losses["G"].append(g_loss_ep / n); losses["D"].append(d_loss_ep / n)
        print(f"[VanillaGAN] Epoch {epoch+1}/{epochs}  "
              f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_dcgan(loader, latent_dim=100, channels=1, epochs=5, device=DEVICE):
    """Trains DCGAN with standard BCE loss."""
    G = DCGANGenerator(latent_dim, channels).to(device)
    D = DCGANDiscriminator(channels).to(device)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    losses = {"G": [], "D": []}

    for epoch in range(epochs):
        g_ep, d_ep = 0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).detach()
            loss_D = criterion(D(imgs), real_lbl) + criterion(D(fake), fake_lbl)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            z = torch.randn(B, latent_dim, device=device)
            loss_G = criterion(D(G(z)), real_lbl)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_ep += loss_G.item(); d_ep += loss_D.item()

        n = len(loader)
        losses["G"].append(g_ep / n); losses["D"].append(d_ep / n)
        print(f"[DCGAN] Epoch {epoch+1}/{epochs}  "
              f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_wgan(loader, latent_dim=100, channels=1, epochs=5,
               n_critic=5, clip=0.01, device=DEVICE):
    """Trains WGAN with weight clipping (Arjovsky et al. 2017)."""
    G = DCGANGenerator(latent_dim, channels).to(device)
    D = WGANDiscriminator(channels).to(device)
    opt_G = optim.RMSprop(G.parameters(), lr=5e-5)
    opt_D = optim.RMSprop(D.parameters(), lr=5e-5)
    losses = {"G": [], "D": []}

    for epoch in range(epochs):
        g_ep, d_ep = 0, 0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            B = imgs.size(0)
            z = torch.randn(B, latent_dim, device=device)
            loss_D = -D(imgs).mean() + D(G(z).detach()).mean()
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            D.clip_weights(clip)
            d_ep += loss_D.item()

            if i % n_critic == 0:
                z = torch.randn(B, latent_dim, device=device)
                loss_G = -D(G(z)).mean()
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()
                g_ep += loss_G.item()

        losses["G"].append(g_ep / max(1, len(loader) // n_critic))
        losses["D"].append(d_ep / len(loader))
        print(f"[WGAN]  Epoch {epoch+1}/{epochs}  "
              f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_wgan_gp(loader, latent_dim=100, channels=1, epochs=5,
                  n_critic=5, lam=10, device=DEVICE):
    """Trains WGAN-GP with gradient penalty (Gulrajani et al. 2017)."""
    G = DCGANGenerator(latent_dim, channels).to(device)
    D = WGANDiscriminator(channels).to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    losses = {"G": [], "D": []}

    for epoch in range(epochs):
        g_ep, d_ep = 0, 0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            B = imgs.size(0)
            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).detach()
            gp = gradient_penalty(D, imgs, fake, device)
            loss_D = -D(imgs).mean() + D(fake).mean() + lam * gp
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            d_ep += loss_D.item()

            if i % n_critic == 0:
                z = torch.randn(B, latent_dim, device=device)
                loss_G = -D(G(z)).mean()
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()
                g_ep += loss_G.item()

        losses["G"].append(g_ep / max(1, len(loader) // n_critic))
        losses["D"].append(d_ep / len(loader))
        print(f"[WGAN-GP] Epoch {epoch+1}/{epochs}  "
              f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_sngan(loader, latent_dim=100, channels=1, epochs=5, device=DEVICE):
    """Trains SN-GAN (spectral normalization on discriminator)."""
    G = DCGANGenerator(latent_dim, channels).to(device)
    D = SNDiscriminator(channels).to(device)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    losses = {"G": [], "D": []}

    for epoch in range(epochs):
        g_ep, d_ep = 0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).detach()
            loss_D = criterion(D(imgs), real_lbl) + criterion(D(fake), fake_lbl)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            z = torch.randn(B, latent_dim, device=device)
            loss_G = criterion(D(G(z)), real_lbl)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_ep += loss_G.item(); d_ep += loss_D.item()

        n = len(loader)
        losses["G"].append(g_ep / n); losses["D"].append(d_ep / n)
        print(f"[SN-GAN] Epoch {epoch+1}/{epochs}  "
              f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_conditional_gan(loader, latent_dim=100, n_classes=10,
                          channels=1, epochs=5, device=DEVICE):
    """Trains Conditional GAN (BigGAN-style label conditioning)."""
    G = ConditionalGenerator(latent_dim, n_classes, channels).to(device)
    D = ConditionalDiscriminator(channels, n_classes).to(device)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    losses = {"G": [], "D": []}

    for epoch in range(epochs):
        g_ep, d_ep = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            z = torch.randn(B, latent_dim, device=device)
            fake = G(z, labels).detach()
            loss_D = (criterion(D(imgs, labels), real_lbl) +
                      criterion(D(fake, labels), fake_lbl))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            z = torch.randn(B, latent_dim, device=device)
            loss_G = criterion(D(G(z, labels), labels), real_lbl)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_ep += loss_G.item(); d_ep += loss_D.item()

        n = len(loader)
        losses["G"].append(g_ep / n); losses["D"].append(d_ep / n)
        print(f"[Cond-GAN] Epoch {epoch+1}/{epochs}  "
              f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_ctgan(X_train, latent_dim=128, epochs=300, batch_size=256, device=DEVICE):
    """Trains CTGAN for tabular data."""
    data_dim = X_train.shape[1]
    G = CTGANGenerator(latent_dim, data_dim).to(device)
    D = CTGANDiscriminator(data_dim).to(device)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    losses = {"G": [], "D": []}

    tensor_X = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(tensor_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        g_ep, d_ep = 0, 0
        for (batch,) in loader:
            B = batch.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).detach()
            loss_D = criterion(D(batch), real_lbl) + criterion(D(fake), fake_lbl)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            z = torch.randn(B, latent_dim, device=device)
            loss_G = criterion(D(G(z)), real_lbl)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            g_ep += loss_G.item(); d_ep += loss_D.item()

        if (epoch + 1) % 50 == 0:
            losses["G"].append(g_ep / len(loader))
            losses["D"].append(d_ep / len(loader))
            print(f"[CTGAN] Epoch {epoch+1}/{epochs}  "
                  f"G: {losses['G'][-1]:.4f}  D: {losses['D'][-1]:.4f}")
    return G, D, losses


def train_timegan(data, latent_dim=24, hidden_dim=24, epochs=200,
                  batch_size=64, device=DEVICE):
    """Trains TimeGAN (Yoon et al. 2019) with supervised + adversarial objectives."""
    n, seq_len, n_features = data.shape
    E  = TimeGANEmbedder(n_features, hidden_dim).to(device)
    R  = TimeGANRecovery(hidden_dim, n_features).to(device)
    Gs = TimeGANGenerator(latent_dim, hidden_dim).to(device)
    Ds = TimeGANDiscriminator(hidden_dim).to(device)

    opt_ER = optim.Adam(list(E.parameters()) + list(R.parameters()), lr=1e-3)
    opt_GD = optim.Adam(list(Gs.parameters()) + list(Ds.parameters()), lr=1e-3)
    criterion = nn.BCELoss()
    mse = nn.MSELoss()
    losses = {"recon": [], "adv": []}

    tensor_data = torch.FloatTensor(data).to(device)
    dataset = TensorDataset(tensor_data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Phase 1: Autoencoder pre-training
    for epoch in range(min(epochs // 2, 100)):
        for (batch,) in loader:
            h = E(batch)
            x_hat = R(h)
            loss = mse(x_hat, batch)
            opt_ER.zero_grad(); loss.backward(); opt_ER.step()
        if (epoch + 1) % 20 == 0:
            losses["recon"].append(loss.item())
            print(f"[TimeGAN-pretraining] Epoch {epoch+1}  Recon: {loss.item():.4f}")

    # Phase 2: Adversarial training
    for epoch in range(epochs // 2):
        for (batch,) in loader:
            B = batch.size(0)
            z = torch.randn(B, seq_len, latent_dim, device=device)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            with torch.no_grad():
                h_real = E(batch)
            h_fake = Gs(z).detach()
            loss_D = criterion(Ds(h_real), real_lbl) + criterion(Ds(h_fake), fake_lbl)
            opt_GD.zero_grad(); loss_D.backward(); opt_GD.step()

            z = torch.randn(B, seq_len, latent_dim, device=device)
            h_fake = Gs(z)
            loss_G = criterion(Ds(h_fake), real_lbl)
            opt_GD.zero_grad(); loss_G.backward(); opt_GD.step()

        if (epoch + 1) % 20 == 0:
            losses["adv"].append(loss_G.item())
            print(f"[TimeGAN-adv] Epoch {epoch+1}  G: {loss_G.item():.4f}  "
                  f"D: {loss_D.item():.4f}")

    return Gs, Ds, E, R, losses


# ─────────────────────────────────────────────
# 8. EVALUATION METRICS (Section 4.3 & 5)
# ─────────────────────────────────────────────

def compute_inception_score(generator, latent_dim, n_samples=1000,
                             splits=10, device=DEVICE):
    """
    Approximate Inception Score using softmax predictions from a simple
    CNN classifier (proxy for Inception v3). Returns mean IS and std.
    IS reflects both quality and diversity (higher = better).
    """
    # Simple proxy classifier
    proxy = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(8),
        nn.Flatten(),
        nn.Linear(32 * 64, 10)
    ).to(device)

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        imgs = generator(z)
        if imgs.dim() == 2:
            imgs = imgs.view(-1, 1, 32, 32)
        logits = proxy(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # IS = exp(E[ KL(p(y|x) || p(y)) ])
    split_size = n_samples // splits
    scores = []
    for i in range(splits):
        p = probs[i * split_size: (i + 1) * split_size]
        p_y = p.mean(axis=0, keepdims=True)
        kl = p * (np.log(p + 1e-8) - np.log(p_y + 1e-8))
        scores.append(np.exp(kl.sum(axis=1).mean()))
    generator.train()
    return float(np.mean(scores)), float(np.std(scores))


def compute_fid(real_features, fake_features):
    """
    Fréchet Inception Distance (Heusel et al. 2017).
    Lower FID indicates closer proximity to real distribution.
    Uses feature activations (not raw pixels).
    """
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def extract_features(generator, latent_dim, n_samples=256, device=DEVICE):
    """Extract flattened activations as proxy image features for FID."""
    feature_extractor = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten()
    ).to(device)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        imgs = generator(z)
        if imgs.dim() == 2:
            imgs = imgs.view(-1, 1, 32, 32)
        feats = feature_extractor(imgs).cpu().numpy()
    generator.train()
    return feats


def tstr_score(G_ctgan, X_real, y_real, latent_dim=128,
               n_synthetic=2000, device=DEVICE):
    """
    Train on Synthetic, Test on Real (TSTR) protocol.
    Trains a logistic regression on synthetic data, evaluates on real.
    Returns accuracy vs real-data ceiling.
    """
    G_ctgan.eval()
    with torch.no_grad():
        z = torch.randn(n_synthetic, latent_dim, device=device)
        X_syn = G_ctgan(z).cpu().numpy()
    G_ctgan.train()

    # Synthetic labels: threshold first feature (mirrors make_tabular_dataset)
    y_syn = (X_syn[:, 1] + X_syn[:, 2] > 0).astype(int)

    clf_syn  = LogisticRegression(max_iter=1000).fit(X_syn, y_syn)
    clf_real = LogisticRegression(max_iter=1000).fit(X_real, y_real)

    # Split real data for evaluation
    split = int(0.8 * len(X_real))
    X_test, y_test = X_real[split:], y_real[split:]

    tstr_acc  = accuracy_score(y_test, clf_syn.predict(X_test))
    trtr_acc  = accuracy_score(y_test, clf_real.predict(X_test))
    print(f"[TSTR] Synthetic-trained acc: {tstr_acc:.4f}  "
          f"Real-data ceiling: {trtr_acc:.4f}  "
          f"Gap: {(trtr_acc - tstr_acc):.4f}")
    return tstr_acc, trtr_acc


def timegan_discriminative_score(Gs, E, real_data, latent_dim=24,
                                 hidden_dim=24, device=DEVICE):
    """
    Discriminative score: post-hoc classifier accuracy on real vs synthetic.
    Score near 0.5 means indistinguishable (ideal); near 1.0 means easily separated.
    """
    n, seq_len, _ = real_data.shape
    Gs.eval(); E.eval()
    with torch.no_grad():
        z_test = torch.randn(n, seq_len, latent_dim, device=device)
        h_fake = Gs(z_test).cpu().numpy()
        h_real = E(torch.FloatTensor(real_data).to(device)).cpu().numpy()

    # Flatten sequences for classifier
    X_real_flat = h_real.reshape(n, -1)
    X_fake_flat = h_fake.reshape(n, -1)
    X = np.vstack([X_real_flat, X_fake_flat])
    y = np.array([1] * n + [0] * n)

    split = int(0.8 * len(X))
    clf = LogisticRegression(max_iter=500).fit(X[:split], y[:split])
    acc = accuracy_score(y[split:], clf.predict(X[split:]))
    disc_score = abs(acc - 0.5)  # 0 = perfect; 0.5 = random
    print(f"[TimeGAN] Discriminative score: {disc_score:.4f}  "
          f"(0=indistinguishable, 0.5=easily detected)")
    return disc_score


# ─────────────────────────────────────────────
# 9. MAIN RUNNER
# ─────────────────────────────────────────────

def run_image_experiments(epochs=3):
    """Runs all image-domain GAN architectures on MNIST."""
    print("\n" + "="*60)
    print("  IMAGE SYNTHESIS EXPERIMENTS (MNIST 32×32)")
    print("="*60)
    loader = get_mnist_loader(batch_size=128)

    results = {}

    G_van, _, l = train_vanilla_gan(loader, epochs=epochs)
    is_mean, is_std = compute_inception_score(G_van, 100)
    print(f"  → Vanilla GAN IS: {is_mean:.3f} ± {is_std:.3f}")
    results["VanillaGAN"] = {"IS": is_mean}

    G_dc, _, l = train_dcgan(loader, epochs=epochs)
    is_mean, is_std = compute_inception_score(G_dc, 100)
    print(f"  → DCGAN IS: {is_mean:.3f} ± {is_std:.3f}")
    results["DCGAN"] = {"IS": is_mean}

    G_wgan, _, l = train_wgan(loader, epochs=epochs)
    is_mean, is_std = compute_inception_score(G_wgan, 100)
    print(f"  → WGAN IS: {is_mean:.3f} ± {is_std:.3f}")
    results["WGAN"] = {"IS": is_mean}

    G_gp, _, l = train_wgan_gp(loader, epochs=epochs)
    is_mean, is_std = compute_inception_score(G_gp, 100)
    print(f"  → WGAN-GP IS: {is_mean:.3f} ± {is_std:.3f}")
    results["WGAN-GP"] = {"IS": is_mean}

    G_sn, _, l = train_sngan(loader, epochs=epochs)
    is_mean, is_std = compute_inception_score(G_sn, 100)
    print(f"  → SN-GAN IS: {is_mean:.3f} ± {is_std:.3f}")
    results["SN-GAN"] = {"IS": is_mean}

    G_cond, _, l = train_conditional_gan(loader, epochs=epochs)
    print(f"  → Conditional GAN (BigGAN-style): training complete")
    results["Cond-GAN"] = {"IS": "N/A (requires label conditioning)"}

    # FID comparison: DCGAN vs WGAN-GP
    real_feats = extract_features(G_dc, 100, n_samples=256)
    fake_feats = extract_features(G_gp, 100, n_samples=256)
    fid = compute_fid(real_feats, fake_feats)
    print(f"\n  FID (DCGAN features vs WGAN-GP features): {fid:.2f}")

    return results


def run_tabular_experiment(epochs=300):
    """Runs CTGAN on synthetic tabular data with TSTR evaluation."""
    print("\n" + "="*60)
    print("  TABULAR SYNTHESIS EXPERIMENT (CTGAN + TSTR)")
    print("="*60)
    X, y = make_tabular_dataset(n=3000)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    G_ct, _, _ = train_ctgan(X_train, epochs=epochs)
    tstr_acc, trtr_acc = tstr_score(G_ct, X_train, y_train.astype(int))
    return {"CTGAN_TSTR": tstr_acc, "Real_ceiling": trtr_acc}


def run_timeseries_experiment(epochs=200):
    """Runs TimeGAN on synthetic time series with discriminative score."""
    print("\n" + "="*60)
    print("  TIME-SERIES SYNTHESIS EXPERIMENT (TimeGAN)")
    print("="*60)
    data = make_timeseries_dataset(n=500, seq_len=24, n_features=5)
    Gs, Ds, E, R, _ = train_timegan(data, epochs=epochs)
    disc = timegan_discriminative_score(Gs, E, data)
    return {"TimeGAN_disc_score": disc}


def summary_table(img_results, tab_results, ts_results):
    """Prints a results summary table mirroring Table 1 in the paper."""
    print("\n" + "="*60)
    print("  RESULTS SUMMARY (cf. Table 1 in the paper)")
    print("="*60)
    print(f"{'Model':<15} {'Domain':<12} {'IS':<10} {'Notes'}")
    print("-" * 60)
    for model, v in img_results.items():
        print(f"{model:<15} {'Image':<12} {str(v.get('IS','—')):<10}")
    tstr = tab_results.get("CTGAN_TSTR", "—")
    ceil = tab_results.get("Real_ceiling", "—")
    print(f"{'CTGAN':<15} {'Tabular':<12} {'N/A':<10} "
          f"TSTR={tstr:.3f}  Ceiling={ceil:.3f}")
    disc = ts_results.get("TimeGAN_disc_score", "—")
    print(f"{'TimeGAN':<15} {'Time-series':<12} {'N/A':<10} "
          f"Disc={disc:.4f} (0=ideal)")
    print("="*60)


if __name__ == "__main__":
    print("GAN Survey Implementation")
    print("Paper: Yadav, Singh, Gehlawat, Garg – Chitkara University\n")

    # Set to fewer epochs for quick demonstration; increase for paper-quality results
    IMAGE_EPOCHS   = 3    # set to 50–200 for publication quality
    TABULAR_EPOCHS = 100  # set to 300+ for CTGAN paper results
    TS_EPOCHS      = 100  # set to 200+ for TimeGAN paper results

    img_res = run_image_experiments(epochs=IMAGE_EPOCHS)
    tab_res = run_tabular_experiment(epochs=TABULAR_EPOCHS)
    ts_res  = run_timeseries_experiment(epochs=TS_EPOCHS)

    summary_table(img_res, tab_res, ts_res)
