"""
Versione script da riga di comando, equivalente a vocal_separation.ipynb

Uso:
    python train.py --dataset_root ./musdb18 --checkpoint_dir ./checkpoints --epochs 20

Su un modello già addestrato, è possibile vedere direttamente le funzioni
evaluate_table() e separate_vocals() in fondo.
"""

import argparse
import os
import time
import gc

import numpy as np
import librosa
import soundfile as sf
import stempeg
import museval
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MUSDBVocalDataset(Dataset):
    """Carica una canzone MUSDB18 alla volta, estrae un segmento casuale di
    lunghezza fissa e calcola spettrogramma del mix + maschera target (IRM)."""

    def __init__(self, root_dir, n_fft=2048, hop_length=512, segment_seconds=4, sr=44100):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.stem.mp4')]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_samples = segment_seconds * sr
        self.sr = sr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stems, rate = stempeg.read_stems(self.files[idx])

        mix = stems[0]
        vocals = stems[4]
        accompaniment = stems[1] + stems[2] + stems[3]

        total_samples = mix.shape[0]
        if total_samples > self.segment_samples:
            start = np.random.randint(0, total_samples - self.segment_samples)
            end = start + self.segment_samples
        else:
            start, end = 0, total_samples

        mix_mono = np.mean(mix[start:end], axis=1)
        vocals_mono = np.mean(vocals[start:end], axis=1)
        accompaniment_mono = np.mean(accompaniment[start:end], axis=1)

        if len(mix_mono) < self.segment_samples:
            pad = self.segment_samples - len(mix_mono)
            mix_mono = np.pad(mix_mono, (0, pad))
            vocals_mono = np.pad(vocals_mono, (0, pad))
            accompaniment_mono = np.pad(accompaniment_mono, (0, pad))

        mix_stft = librosa.stft(mix_mono, n_fft=self.n_fft, hop_length=self.hop_length)
        vocals_stft = librosa.stft(vocals_mono, n_fft=self.n_fft, hop_length=self.hop_length)
        accompaniment_stft = librosa.stft(accompaniment_mono, n_fft=self.n_fft, hop_length=self.hop_length)

        mix_mag = np.abs(mix_stft)
        vocals_mag = np.abs(vocals_stft)
        accompaniment_mag = np.abs(accompaniment_stft)

        epsilon = 1e-10
        mask = vocals_mag / (vocals_mag + accompaniment_mag + epsilon)

        mix_tensor = torch.from_numpy(mix_mag).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return mix_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Modello: U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Encoder-decoder con skip connections. Input: spettrogramma del mix.
    Output: maschera (0-1) da applicare al mix per isolare la voce."""

    def __init__(self):
        super().__init__()
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(128, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(32, 16)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self._match_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_cat(self.up1(d2), e1))
        return torch.sigmoid(self.out(d1))

    def _match_and_cat(self, upsampled, skip):
        diffY = skip.size(2) - upsampled.size(2)
        diffX = skip.size(3) - upsampled.size(3)
        upsampled = nn.functional.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                                     diffY // 2, diffY - diffY // 2])
        return torch.cat([upsampled, skip], dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_loader, test_loader, device, checkpoint_dir, num_epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Ripreso dal checkpoint, riparto dall'epoca {start_epoch}")

    train_losses, val_losses = [], []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for batch_mix, batch_mask in train_loader:
            batch_mix, batch_mask = batch_mix.to(device), batch_mask.to(device)

            optimizer.zero_grad()
            pred_mask = model(batch_mix)
            loss = criterion(pred_mask, batch_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_mix, batch_mask in test_loader:
                batch_mix, batch_mask = batch_mix.to(device), batch_mask.to(device)
                pred_mask = model(batch_mix)
                val_running_loss += criterion(pred_mask, batch_mask).item()
        val_loss = val_running_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoca {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f} - "
              f"val_loss: {val_loss:.4f} - tempo: {time.time()-epoch_start:.1f}s")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Inferenza e valutazione
# ---------------------------------------------------------------------------

def separate_vocals(model, mix_audio, n_fft=2048, hop_length=512, device='cpu'):
    """Applica il modello a un audio mono e restituisce la voce stimata."""
    mix_stft = librosa.stft(mix_audio, n_fft=n_fft, hop_length=hop_length)
    mix_mag = np.abs(mix_stft)
    mix_phase = np.angle(mix_stft)

    mix_tensor = torch.from_numpy(mix_mag).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_mask = model(mix_tensor).squeeze().cpu().numpy()

    estimated_vocals_mag = pred_mask * mix_mag
    estimated_vocals_stft = estimated_vocals_mag * np.exp(1j * mix_phase)
    return librosa.istft(estimated_vocals_stft, hop_length=hop_length)


def sdr_of(ref, est):
    """SDR mediano (dB) tra voce di riferimento e voce stimata."""
    n = min(len(ref), len(est))
    sdr, _, _, _ = museval.evaluate(ref[:n].reshape(1, -1, 1), est[:n].reshape(1, -1, 1))
    return np.nanmedian(sdr)


def evaluate_table(model, test_path, n_songs=10, excerpt_sec=30, start_sec=30,
                   n_fft=2048, hop_length=512, device='cpu'):
    """Per ogni brano (un estratto di excerpt_sec secondi): SDR di baseline
    (il mix come "voce"), modello e oracolo (maschera ideale applicata al mix)."""
    files = sorted(f for f in os.listdir(test_path) if f.endswith('.stem.mp4'))[:n_songs]
    results = []

    for fname in files:
        stems, rate = stempeg.read_stems(os.path.join(test_path, fname))
        total = stems.shape[1]
        s = min(start_sec * rate, max(0, total - excerpt_sec * rate))
        e = min(s + excerpt_sec * rate, total)

        mix = stems[0][s:e].mean(axis=1)
        voc = stems[4][s:e].mean(axis=1)
        acc = (stems[1] + stems[2] + stems[3])[s:e].mean(axis=1)
        del stems
        gc.collect()

        base = sdr_of(voc, mix)

        stft = lambda x: librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
        M, V, A = stft(mix), stft(voc), stft(acc)
        irm = np.abs(V) / (np.abs(V) + np.abs(A) + 1e-10)
        oracle = sdr_of(voc, librosa.istft(irm * M, hop_length=hop_length))

        mod = sdr_of(voc, separate_vocals(model, mix, n_fft=n_fft, hop_length=hop_length, device=device))

        results.append((base, mod, oracle))
        print(f"{fname[:40]:40s} baseline {base:6.2f} | modello {mod:6.2f} | oracolo {oracle:6.2f}")

    res = np.array(results)
    print("\n--- MEDIANA sulle canzoni (dB) ---")
    print(f"Baseline: {np.median(res[:,0]):.2f}   Modello: {np.median(res[:,1]):.2f}   Oracolo: {np.median(res[:,2]):.2f}")
    print("--- MEDIA sulle canzoni (dB) ---")
    print(f"Baseline: {res[:,0].mean():.2f}   Modello: {res[:,1].mean():.2f}   Oracolo: {res[:,2].mean():.2f}")
    print(f"Canzoni in cui il modello batte la baseline: {(res[:,1] > res[:,0]).sum()}/{len(res)}")
    return res


def separate_file(model, input_path, output_path, device='cpu'):
    """Carica un file audio qualsiasi, isola la voce e salva il risultato."""
    audio, sr = librosa.load(input_path, sr=44100, mono=True)
    vocals = separate_vocals(model, audio, device=device)
    sf.write(output_path, vocals, sr)
    print(f"Voce isolata salvata in: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training e valutazione del modello di vocal separation")
    parser.add_argument('--dataset_root', type=str, required=True, help="Cartella con dentro train/ e test/")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--segment_seconds', type=int, default=4)
    parser.add_argument('--eval_songs', type=int, default=10, help="Numero di canzoni test su cui calcolare SDR (un estratto da 30 s ciascuna)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_path = os.path.join(args.dataset_root, 'train')
    test_path = os.path.join(args.dataset_root, 'test')

    train_dataset = MUSDBVocalDataset(train_path, segment_seconds=args.segment_seconds)
    test_dataset = MUSDBVocalDataset(test_path, segment_seconds=args.segment_seconds)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Canzoni train: {len(train_dataset)} - Canzoni test: {len(test_dataset)}")

    model = UNet().to(device)
    print(f"Numero parametri modello: {sum(p.numel() for p in model.parameters())}")

    train_losses, val_losses = train(
        model, train_loader, test_loader, device,
        checkpoint_dir=args.checkpoint_dir, num_epochs=args.epochs, lr=args.lr
    )

    print("\nValutazione SDR (estratto di 30 s per brano):")
    evaluate_table(model, test_path, n_songs=args.eval_songs, device=device)


if __name__ == '__main__':
    main()
