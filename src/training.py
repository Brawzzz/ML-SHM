#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import tqdm
import copy
import torch
import torch.optim as optim
import numpy as np 

from torch.utils.data import DataLoader, TensorDataset

import tools

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
#================================================================================#
def train_model(model_cls    : type[BaseDetector], 
                params       : dict, 
                data         : dict, 
                train_config : dict, 
                seed         : int, 
                device,
                on_epoch_end        = None,
                verbose      : bool = True) -> tuple[BaseDetector, dict]:

    """
    Build and train a new model from `params` 

    Parameters:
    ----------
    model_cls    : (type[BaseDetector]) model class (e.g. ConvAutoEncoder)
    params       : (dict) complete hyperparameters (architecture + training parameters)
    data         : (dict) prepared data {"train": (X, y), "val": (X, y), "test": (X, y)}
    train_config : (dict) "training" section of the config (epochs, patience, num_workers)
    seed         : (int) random seed for this training
    device       : (torch.device) cpu or cuda
    verbose      : (bool) whether to show a progress bar -- defaults True

    Returns:
    ----------
    (model, history) : (tuple) trained model and training history (train + validation loss)
    """

    #---------------------------------------------
    tools.set_seed(seed)

    (X_val, y_val)               = data["validation"]
    (archi_params, train_params) = model_cls.split_params(params)
    
    model = model_cls(input_length=X_val.shape[-1], in_channels=X_val.shape[1], **archi_params)
    
    history = fit(model       = model,
                  train       = data["train"],
                  validation  = (X_val[y_val == 0], y_val[y_val == 0]),
                  **train_params,
                  epochs       = train_config["epochs"],
                  patience     = train_config["patience"],
                  device       = device,
                  num_workers  = train_config["num_workers"],
                  seed         = seed,
                  on_epoch_end = on_epoch_end,
                  verbose      = verbose)

    return (model, history)

#================================================================================#
def fit(model        : BaseDetector,
        train        : tuple,
        validation   : tuple,
        *,
        lr           : float,
        batch_size   : int,
        optimizer    : str          = "adam",
        weight_decay : float        = 0.0,
        epochs       : int          = 100,
        patience     : int          = 10,
        min_delta    : float        = 0.0,
        device       : torch.device = torch.device("cpu"),
        num_workers  : int          = 0,
        seed         : int          = None,
        on_epoch_end                = None,
        verbose      : bool         = True) -> dict:

    """
    Generic training and validation loop for a BaseDetector model.

    The model defines its own loss (model.training_step), the loop only handles
    optimisation, validation, early stopping and computes the best weights.

    Parameters
    ----------
    model        : BaseDetector to train (moved to `device`)
    train        : (X, y) training set
    validation   : (X, y) validation set used for the validation loss and early stopping
    
    lr | batch_size | optimizer | weight_decay : optimisation hyperparameters

    epochs       : (int) maximum number of epochs
    patience     : (int) number of epochs without improvement for early stopping | if <= 0 -> disabled -- defaults : patience = 10
    min_delta    : (float) minimum decrease of the validation loss to count as an improvement
    device       : (torch.device) device to perform training -- defaults device = "cpu"
    num_workers  : (int) DataLoader workers -- defaults : num_workers = 0
    seed         : (int) seed of the training batches order

    on_epoch_end : optional callback(epoch, model, val_loss), e.g. Optuna pruning
                   (may raise optuna.TrialPruned, which is propagated)
    verbose      : (bool) whether to show a progress bar -- defaults : verbose = True

    Examples:
    ----------
        - fit(model = ConvAutoEncoder,
              train = )
              
    Returns
    ----------
    history : (dict) {"train_loss": [], "val_loss": [], "best_epoch": int, "best_val_loss": float}
    """

    #---------------------------------------------
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer : {optimizer} (available : {list(OPTIMIZERS)})")

    optim = OPTIMIZERS[optimizer](model.parameters(), lr=lr, weight_decay=weight_decay)

    #---------------------------------------------
    (X_train, y_train) = train
    (X_val, y_val)     = validation

    train_loader = _loader(X           = X_train,
                           y           = y_train, 
                           batch_size  = batch_size, 
                           shuffle     = True, 
                           num_workers = num_workers, 
                           seed        = seed)
    
    val_loader = _loader(X           = X_val,
                         y           = y_val,
                         batch_size  = max(batch_size, 256), 
                         shuffle     = False, 
                         num_workers = num_workers)

    #---------------------------------------------
    history    = {"train_loss": [], "val_loss": [], "best_epoch": 0, "best_val_loss": float("inf")}
    best_state = copy.deepcopy(model.state_dict())
    impatience = 0
    
    pbar = tqdm.tqdm(range(epochs), desc=f"Training {model.NAME}", unit="epoch", disable=not verbose)
    
    model.to(device)

    for epoch in pbar:

        #------------------------- TRAIN -------------------------#
        model.train()
        train_loss = 0.0

        for batch in train_loader:

            batch = [b.to(device, non_blocking=True) for b in batch]

            optim.zero_grad()

            loss = model.training_step(batch)
            loss.backward()

            optim.step()

            train_loss += loss.item() * len(batch[0])

        train_loss /= len(train_loader.dataset)

        #------------------------- VALIDATION ---------------------#
        val_loss = validation_loss(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        pbar.set_postfix({"train": f"{train_loss:.3e}", "val": f"{val_loss:.3e}"})

        #------------------------------
        if val_loss < history["best_val_loss"] - min_delta:

            history["best_epoch"]    = epoch
            history["best_val_loss"] = val_loss

            impatience = 0
            best_state = copy.deepcopy(model.state_dict())

        else:
            impatience += 1

        #------------------------------
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, val_loss)

        if patience > 0 and impatience >= patience:
            break

    pbar.close()

    #---------------------------------------------
    model.load_state_dict(best_state)

    return history

#================================================================================#
def _loader(X           : np.ndarray, 
            y           : np.ndarray, 
            batch_size  : int, 
            shuffle     : bool, 
            num_workers : int = 0, 
            seed        : int = None) -> DataLoader:

    """
    DataLoader returning (x, y) batches, x of shape (batch, 1, signal_length)

    Parameters
    ----------
    X           : (np.ndarray) input signals - shape (n, input_len)
    y           : (np.ndarray) signal's labels - healty : 0 | damaged : 1 - shape (n,)
    batch_size  : (int) number of signals in one batch 
    shuffle     : (bool) Whether to shuffle signals for each epoch 
    num_workers : (int) number of parallel process foe batches preparation -- defaults : num_workers = 0
    seed        : (int) random generator initialization, allows to reproduce the shuffle -- defaults : seed = None
                        if None no generator is build
    """

    #---------------------------------------------
    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)

    y_tensor = torch.as_tensor(y, dtype=torch.long)

    #--------------------
    dataset   = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    return DataLoader(dataset     = dataset, 
                      batch_size  = batch_size, 
                      shuffle     = shuffle, 
                      num_workers = num_workers,
                      generator   = generator, 
                      pin_memory  = torch.cuda.is_available())

#================================================================================#
@torch.no_grad()
def validation_loss(model  : BaseDetector,
                    loader : DataLoader,
                    device : torch.device) -> float :

    """
    Mean validation loss of the model over a DataLoader

    Parameters
    ----------
    model  : (BaseDetector) trained model
    loader : (DataLoader) dataloader setup for validation set 
    device : (torch.device) device 

    Returns
    ----------
    val_loss : (float) model's validation loss over the loader 
    """

    #---------------------------------------------
    model.eval()
    (total, n) = (0.0, 0)

    for batch in loader:

        batch  = [b.to(device, non_blocking=True) for b in batch]
        total += model.validation_loss(batch).item() * len(batch[0])
        n     += len(batch[0])

    val_loss = total / max(n, 1)
    
    return val_loss

#================================================================================#
@torch.no_grad()
def predict_scores(model      : BaseDetector, 
                   X          : np.ndarray, 
                   device     : torch.device, 
                   batch_size : int = 256) -> np.ndarray:

    """
    Anomaly score of every signal of `X`

    Parameters
    ----------
    model      : (BaseDetector)
    X          : (np.ndarray)
    decvice    : (torch.device) cpu or cuda
    batch_size : (int) size of a single batch

    Returns
    ----------
    scores : (n_signals,) array of anomaly scores
    """

    #---------------------------------------------
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)

    model.eval()
    model.to(device)

    #-------------------
    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)

    #-------------------
    scores = []

    for i in range(0, len(X_tensor), batch_size):

        eval = model.anomaly_score(X_tensor[i : i + batch_size].to(device)).cpu()
        scores.append(eval)

    return torch.cat(scores).numpy().astype(np.float64)

