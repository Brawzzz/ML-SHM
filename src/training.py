#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import copy

import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models import BaseDetector


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
OPTIMIZERS = {
    "adam"  : optim.Adam,
    "adamw" : optim.AdamW,
    "sgd"   : optim.SGD,
}

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                num_workers: int = 0, seed: int = None) -> DataLoader:

    """
    DataLoader yielding (x, y) batches, x of shape (batch, 1, signal_length)

    num_workers = 0 is recommended on Windows : the data is already in memory and
    worker processes are expensive to start there.
    """

    #---------------------------------------------
    dataset   = TensorDataset(torch.as_tensor(X, dtype=torch.float32).unsqueeze(1),
                              torch.as_tensor(y, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=generator, pin_memory=torch.cuda.is_available())

#================================================================================#
def fit(model: BaseDetector,
        train: tuple,
        val: tuple,
        *,
        lr: float,
        batch_size: int,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.0,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
        seed: int = None,
        on_epoch_end=None,
        verbose: bool = True) -> dict:

    """
    Generic training loop, identical for every BaseDetector.

    The model defines its own loss (model.training_step), the loop only handles
    optimisation, validation, early stopping and the restoration of the best weights.

    Parameters
    ----------
    model        : BaseDetector to train (moved to `device`)
    train        : (X, y) training set
    val          : (X, y) validation set used for the validation loss / early stopping
    lr, batch_size, optimizer, weight_decay : optimisation hyperparameters
    epochs       : maximum number of epochs
    patience     : early stopping patience (epochs without improvement), <= 0 -> disabled
    min_delta    : minimum decrease of the validation loss to count as an improvement
    device       : torch device
    num_workers  : DataLoader workers
    seed         : seed of the training batches order
    on_epoch_end : optional callback(epoch, model, val_loss), e.g. Optuna pruning
                   (may raise optuna.TrialPruned, which is propagated)
    verbose      : show a progress bar

    Returns
    ----------
    history : {"train_loss": [...], "val_loss": [...], "best_epoch": int, "best_val_loss": float}
    """

    #---------------------------------------------
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer : {optimizer} (available : {list(OPTIMIZERS)})")

    model.to(device)

    train_loader = make_loader(*train, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed)
    val_loader   = make_loader(*val,   batch_size=max(batch_size, 256), shuffle=False, num_workers=num_workers)

    opt = OPTIMIZERS[optimizer](model.parameters(), lr=lr, weight_decay=weight_decay)

    #---------------------------------------------
    history    = {"train_loss": [], "val_loss": [], "best_epoch": 0, "best_val_loss": float("inf")}
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    pbar = tqdm(range(epochs), desc=f"Training {model.NAME}", unit="epoch", disable=not verbose)

    for epoch in pbar:

        #------------------------------ train
        model.train()
        train_loss = 0.0

        for batch in train_loader:

            batch = [b.to(device, non_blocking=True) for b in batch]

            opt.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            opt.step()

            train_loss += loss.item() * len(batch[0])

        train_loss /= len(train_loader.dataset)

        #------------------------------ validation
        val_loss = evaluate_loss(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        pbar.set_postfix({"train": f"{train_loss:.3e}", "val": f"{val_loss:.3e}"})

        #------------------------------ best weights / early stopping
        if val_loss < history["best_val_loss"] - min_delta:

            history["best_val_loss"] = val_loss
            history["best_epoch"]    = epoch
            best_state               = copy.deepcopy(model.state_dict())
            wait                     = 0

        else:
            wait += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch, model, val_loss)

        if patience > 0 and wait >= patience:
            break

    pbar.close()

    #---------------------------------------------
    model.load_state_dict(best_state)

    return history

#================================================================================#
@torch.no_grad()
def evaluate_loss(model: BaseDetector, loader: DataLoader, device: torch.device) -> float:

    """Mean validation loss of the model over a DataLoader"""

    #---------------------------------------------
    model.eval()
    (total, n) = (0.0, 0)

    for batch in loader:

        batch  = [b.to(device, non_blocking=True) for b in batch]
        total += model.validation_loss(batch).item() * len(batch[0])
        n     += len(batch[0])

    return total / max(n, 1)

#================================================================================#
@torch.no_grad()
def predict_scores(model: BaseDetector, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:

    """
    Anomaly score of every signal of X (batched to avoid VRAM saturation)

    Returns
    ----------
    scores : (n_signals,) array, the higher the more anomalous
    """

    #---------------------------------------------
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)

    model.eval()
    model.to(device)

    tensor = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    scores = []

    for i in range(0, len(tensor), batch_size):
        scores.append(model.anomaly_score(tensor[i : i + batch_size].to(device)).cpu())

    return torch.cat(scores).numpy().astype(np.float64)
