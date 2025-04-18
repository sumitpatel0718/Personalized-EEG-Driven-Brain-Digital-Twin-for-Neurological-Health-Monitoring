# data_loader.py
import mne
import numpy as np
import os

class EEGDataLoader:
    """
    Loads raw EEG data from .edf files and preprocesses it for model input.
    """
    def __init__(self, data_dir: str, montage: str = 'standard_1020'):
        self.data_dir = data_dir
        self.montage = montage

    def load_subject(self, subject_id: str) -> mne.io.BaseRaw:
        edf_path = os.path.join(self.data_dir, f"{subject_id}.edf")
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.set_montage(self.montage)
        raw.filter(1., 40.)
        return raw

    def epoch_data(self, raw: mne.io.BaseRaw, event_id: dict, tmin: float, tmax: float):
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=(None, 0), preload=True)
        return epochs.get_data()


# model_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class EEGTransformer(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, d_model: int = 128, n_heads: int = 8):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_proj = nn.Linear(n_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(d_model, 2)  # binary: normal vs. event

    def forward(self, x):  # x: [batch, seq_len, n_channels]
        x = self.input_proj(x)
        x = x + self.positional_encoding
        x = x.permute(1, 0, 2)  # seq_len, batch, d_model
        out = self.encoder(x)
        out = out.mean(dim=0)
        return self.classifier(out)


def train_model(train_X, train_y, val_X, val_y, epochs=30, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGTransformer(n_channels=train_X.shape[2], seq_len=train_X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_y))
    val_ds   = TensorDataset(torch.Tensor(val_X), torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # (Optional) add validation loop, logging, early stopping
    return model


# simulation_engine.py
import numpy as np

def simulate_intervention(model, baseline_epoch: np.ndarray, changed_pattern: np.ndarray) -> np.ndarray:
    """
    Apply a simulated perturbation (e.g., drug effect) to the EEG epoch and predict outcome.
    """
    # simple additive perturbation for demo
    perturbed = baseline_epoch + changed_pattern
    with torch.no_grad():
        pred = model(torch.Tensor(perturbed[None, ...]))
        probs = nn.Softmax(dim=-1)(pred)
    return probs.numpy()


# main.py
from data_loader import EEGDataLoader
from model_training import train_model, EEGTransformer
from simulation_engine import simulate_intervention
import numpy as np

if __name__ == '__main__':
    # Dataset link: https://physionet.org/content/chbmit/1.0.0/
    DATA_DIR = '/path/to/chbmit'  # download .edf files here
    loader = EEGDataLoader(DATA_DIR)
    raw = loader.load_subject('chb01_03')
    X = loader.epoch_data(raw, event_id={'seizure': 1}, tmin=-1., tmax=2.)
    y = np.array([1 if ev else 0 for ev in raw.annotations.describe()])

    # Split train/val manually or with sklearn
    model = train_model(X[:50], y[:50], X[50:60], y[50:60])

    # Simulate a perturbation
    baseline = X[0]
    drug_effect = np.random.normal(scale=0.1, size=baseline.shape)
    result_probs = simulate_intervention(model, baseline, drug_effect)
    print(f"Predicted probabilities post-intervention: {result_probs}")
