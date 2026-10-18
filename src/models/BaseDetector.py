#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import torch
import torch.nn as nn

from typing import Self


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
ACTIVATIONS = {
    
    "relu"       : nn.ReLU, 
    "leaky_relu" : nn.LeakyReLU, 
    "gelu"       : nn.GELU, 
    "elu"        : nn.ELU
}

#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class BaseDetector(nn.Module):
    
	"""
	Common interface of every PyTorch anomaly detector.

	A model OWNS its hyperparameters :

		- DEFAULTS     : value of every hyperparameter
		- SEARCH_SPACE : how each tunable hyperparameter is sampled by Optuna

	The tuned values are NOT stored in the model or in the config : they are an
	OUTPUT of the tuning, saved (model, dataset) in ./results/<experiment>/best_params.json.

	A subclass implements :

		- __init__(input_length, **architecture_params)
		- forward(x)
		- training_step(batch)  -> scalar loss minimised by the generic training loop
		- anomaly_score(x)      -> one score per signal -- describe the anomaly rate

	SEARCH_SPACE example :
	
		{
			"n_layers"      : [2, 3, 4],
			"base_channels" : [8, 16, 32],
			"kernel_size"   : [3, 5, 7, 9],
			"decode_method" : ["resize", "transpose"],
			"activation"    : ["relu", "leaky_relu", "gelu"],
			"lr"            : [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
			"batch_size"    : [16, 32, 64],
		}
    """
  
	#---------------------------------------------
	NAME          : str  = "base"
	DEFAULTS      : dict = {}
	SEARCH_SPACE  : dict = {}

	TRAINING_KEYS = ("lr", "batch_size", "optimizer", "weight_decay")

	#================================================================================#
	def __init__(self, input_length: int, in_channels: int = 1):

		super().__init__()
		self.input_length = input_length
		self.in_channels  = in_channels

	#================================================================================#
	def training_step(self, batch: list) -> torch.Tensor:
		raise NotImplementedError

	#================================================================================#
	def validation_loss(self, batch: list) -> torch.Tensor:
		
		"""
		Loss on a validation batch, same as training by default

		Args
		----------
		batch : (list) 

		Returns
		----------
		tensor : (torch.Tensor) 

		"""
		return self.training_step(batch)

	#================================================================================#
	def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	#================================================================================#
	@classmethod
	def resolve_params(cls, params: dict = None) -> dict:
        
		"""
		Update the value of DEFAULTS with the values in `params`. 
		Unknown names raise an error

		Args
		----------
		params : (dict) dictionary containing the params to complete / update DEFAULTS

		Return
		----------
		complete_params : (dict) dictonart containing the all parameters (models_cls.DEFAULTS + params)
		"""

		#---------------------------------------------
		params  = dict(params or {})
		unknown = set(params) - set(cls.DEFAULTS)

		if unknown:
			raise ValueError(f"[{cls.NAME}] unknown hyperparameter(s) : {sorted(unknown)} "
			                 f"(available : {sorted(cls.DEFAULTS)})")
		
		complete_params = {**cls.DEFAULTS, **params}

		return complete_params

	#================================================================================#
	@classmethod
	def split_params(cls, params: dict) -> tuple[dict, dict]:

		"""
		Separate the models and the training hyperparameters 

		Args
		----------
		params : (dict) dictionnary contaning the hyperparameters to split

		Returns
		----------
		hp_split : (tuple) contain the model and the training hyperparameters (models_hp, training_hp)

		Example
		----------

		"""

		#---------------------------------------------
		model_hp = {k : v for k, v in params.items() if k not in cls.TRAINING_KEYS}
		train_hp = {k : v for k, v in params.items() if k in cls.TRAINING_KEYS}

		hp_split = (model_hp, train_hp)

		return(hp_split)

	#================================================================================#
	@classmethod
	def build(cls, input_len, params=None, in_channels: int = 1) -> Self:

		"""
		Instanciate a new model from a paremeter dictionnary

		Args
		----------
		input_len 	: (int) length of the input signal
		params	  	: (dict) dictionnary containing the wanted paremeters config -- defaults params = None
					if params = None the new model is build from DEFAULTS
		in_channels : (int)

		Returns
		----------
		model : (BaseDetector) new model 
		"""

		#---------------------------------------------

		params         = cls.resolve_params(params)
		(models_hp, _) = cls.split_params(params)

		return cls(input_len, in_channels=in_channels, **models_hp)

	
		