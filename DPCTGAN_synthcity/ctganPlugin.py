"""
Reference: "Differentially Private Generative Adversarial Network", Xie, Liyang  et al.
"""

# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from torch.utils.data import sampler
import torch

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_gan import TabularGAN
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE

from plugin import Plugin
from protectionLevel import ProtectionLevel

# SDV
from ctgan import CTGAN

import os


class CTGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.privacy.plugin_dpgan.DPGANPlugin
        :parts: 1

    Differentially Private Generative Adversarial Network implementation. The discriminator is trained using DP-SGD.

    Args:
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        n_iter: int
            Maximum number of iterations in the Generator.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        random_state: int
            random seed to use
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before early stopping is trigged.
        patience_metric: Optional[WeightedMetrics]
            If not None, the metric is used for evaluation the criterion for early stopping.
        # privacy settings
        dp_enabled: bool
            Train the discriminator with Differential Privacy guarantees
        dp_delta: Optional[float]
            Optional DP delta: the probability of information accidentally being leaked. Usually 1 / len(dataset)
        dp_epsilon: float = 3
            DP epsilon: privacy budget, which is a measure of the amount of privacy that is preserved by a given algorithm. Epsilon is a number that represents the maximum amount of information that an adversary can learn about an individual from the output of a differentially private algorithm. The smaller the value of epsilon, the more private the algorithm is. For example, an algorithm with an epsilon of 0.1 preserves more privacy than an algorithm with an epsilon of 1.0.
        dp_max_grad_norm: float
            max grad norm used for gradient clipping
        dp_secure_mode: bool = False,
             if True uses noise generation approach robust to floating point arithmetic attacks.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("dpgan", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        protection_level: ProtectionLevel = None,
        discrete_columns: List[str] = [],
        embedding_dim=128, 
        generator_dim=(256, 256),
        discriminator_dim=(256, 256), 
        generator_lr=2e-4, 
        generator_decay=1e-6,
        discriminator_lr=2e-4, 
        discriminator_decay=1e-6, 
        batch_size=500,
        discriminator_steps=1, 
        log_frequency=True, 
        verbose=False, 
        epochs=300,
        pac=10, 
        cuda=True,
        # core plugin arguments
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            protection_level=protection_level,
            **kwargs
        )

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda
        }

        self.discrete_columns = discrete_columns

        self.cwd = os.getcwd()
        self.directory = os.path.join("generators", "DPGAN")

        self.saved = False

    @staticmethod
    def name() -> str:
        return "dpgan"

    @staticmethod
    def type() -> str:
        return "privacy"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "CTGANPlugin":
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = kwargs["cond"]
        
        self.model = CTGAN(**self._model_kwargs)
        self.model.fit(X.dataframe(), discrete_columns=self.discrete_columns)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = kwargs["cond"]

        return GenericDataLoader(self.model.sample(count))
    
    def save_model(self):
        ...
    
    def load_model(self, path):
        ...
    
    def load_protection_level(self, file_path):
        ...
    
    def find_models_by_protection_level(self, target_protection_level):
        ...
