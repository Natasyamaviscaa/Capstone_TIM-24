from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
NIM = "123450011"
OUT = ROOT / "results" / NIM
MODEL_DIR = ROOT / "models" / NIM


class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.fc(self.dropout(hidden[-1]))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_x(x, mean=None, std=None):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = x[:, :, None]
    elif x.ndim != 3:
        raise ValueError(
            f"Input harus berbentuk [N, fitur] atau [N, langkah, fitur], "
            f"bukan {x.shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError("Input mengandung NaN atau nilai tak hingga.")

    if mean is None:
        mean = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
        std[std < 1e-8] = 1.0

    return (x - mean) / std, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(ROOT))

    try:
        from src.protokol.data import muat
        X_train, y_train, X_val, y_val, X_test, y_test = muat()
    except NotImplementedError as exc:
        raise SystemExit(
            "\nDATA LOADER BELUM DIISI.\n"
            "src/protokol/data.py masih berupa template. "
            "Training belum dapat dilakukan secara valid.\n"
            f"Detail: {exc}"
        )

    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.int64).reshape(-1)

    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Jumlah data dan label tidak sama.")

    classes = np.unique(y_train)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Label kelas harus berupa bilangan 0 sampai K-1.")

    if not set(np.unique(y_val)).issubset(set(classes)):
        raise ValueError("Label validasi memiliki kelas yang tidak ada di data latih.")

    X_train, mean, std = prepare_x(X_train)
    X_val, _, _ = prepare_x(X_val, mean, std)

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Dimensi data latih dan validasi tidak sama.")

    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = RNNClassifier(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_classes=len(classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    patience = 8
    waiting = 0

    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(val_tensor).argmax(dim=1).cpu().numpy()

        score = f1_score(y_val, pred, average="macro", zero_division=0)
        print(f"Epoch {epoch + 1:02d} | val_macro_f1={score:.5f}")

        if score > best_f1:
            best_f1 = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1

        if waiting >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan model.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_val = model(val_tensor).argmax(dim=1).cpu().numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUT / "pred_val.npy", pred_val)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_mean": mean,
            "input_std": std,
            "classes": classes,
            "seed": args.seed,
        },
        MODEL_DIR / "model.pt",
    )

    print("\nEvaluasi validasi:")
    print("Accuracy :", accuracy_score(y_val, pred_val))
    print("Macro F1 :", f1_score(y_val, pred_val, average="macro"))
    print("Prediksi :", OUT / "pred_val.npy")
    print("Durasi   :", round(time.time() - start, 2), "detik")

    if args.final and X_test is not None:
        if y_test is None:
            raise ValueError("Data uji tersedia tetapi label uji tidak tersedia.")
        X_test, _, _ = prepare_x(X_test, mean, std)
        with torch.no_grad():
            pred_test = model(
                torch.tensor(X_test, dtype=torch.float32).to(device)
            ).argmax(dim=1).cpu().numpy()
        np.save(OUT / "pred_test.npy", pred_test)
        print("Prediksi uji:", OUT / "pred_test.npy")


if __name__ == "__main__":
    main()
