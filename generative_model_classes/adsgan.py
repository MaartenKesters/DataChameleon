"""
References: 
- Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
        "Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
        A harmonizing advancement for AI in medicine,"
        IEEE Journal of Biomedical and Health Informatics (JBHI), 2019. (https://ieeexplore.ieee.org/document/9034117)
- "Synthcity: facilitating innovative use cases of synthetic data in different data modalities", Zhaozhi, Qian et al. (https://arxiv.org/abs/2301.07573)
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
import os

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.tabular_gan import TabularGAN
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE

from generative_model_classes.plugin import Plugin
from protectionLevel import ProtectionLevel

class AdsGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.privacy.plugin_adsgan.AdsGANPlugin
        :parts: 1

    AdsGAN plugin - Anonymization through Data Synthesis using Generative Adversarial Networks.

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
        adjust_inference_sampling: bool
            Adjust the marginal probabilities in the synthetic data to closer match the training set. Active only with the ConditionalSampler
        lambda_gradient_penalty: float = 10
            Weight for the gradient penalty
        lambda_identifiability_penalty: float = 0.1
            Weight for the identifiability penalty, if enabled
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before training early stopping is trigged.
        patience_metric: Optional[WeightedMetrics]
            If not None, the metric is used for evaluation the criterion for training early stopping.
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
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("adsgan", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        protection_level: ProtectionLevel = None,
        n_iter: int = 10000,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 500,
        generator_nonlin: str = "relu",
        generator_dropout: float = 0.1,
        generator_opt_betas: tuple = (0.5, 0.999),
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 500,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        discriminator_opt_betas: tuple = (0.5, 0.999),
        # training
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 200,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        encoder_max_clusters: int = 5,
        encoder: Any = None,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        device: Any = DEVICE,
        adjust_inference_sampling: bool = False,
        # privacy settings
        epsilon: float = 1,
        delta: Optional[float] = None,
        dp_max_grad_norm: float = 2,
        dp_secure_mode: bool = False,
        # early stopping
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        # core plugin arguments
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            protection_level=protection_level,
            **kwargs
        )
        if patience_metric is None:
            patience_metric = WeightedMetrics(
                metrics=[("detection", "detection_mlp")],
                weights=[1],
                workspace=workspace,
            )
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.n_iter = n_iter
        self.generator_dropout = generator_dropout
        self.generator_opt_betas = generator_opt_betas
        self.generator_extra_penalties = ["identifiability_penalty"]
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_opt_betas = discriminator_opt_betas

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_identifiability_penalty = lambda_identifiability_penalty

        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.dataloader_sampler = dataloader_sampler

        self.device = device
        self.patience = patience
        self.patience_metric = patience_metric
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.adjust_inference_sampling = adjust_inference_sampling

        # privacy
        self.dp_epsilon = epsilon
        self.dp_delta = delta
        self.dp_enabled = True
        self.dp_max_grad_norm = dp_max_grad_norm
        self.dp_secure_mode = dp_secure_mode

        self.cwd = os.getcwd()
        self.directory = os.path.join("generators", "ADSGAN")

    @staticmethod
    def name() -> str:
        return "adsgan"

    @staticmethod
    def type() -> str:
        return "privacy"

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "AdsGANPlugin":
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = kwargs["cond"]

        self.model = TabularGAN(
            X.dataframe(),
            cond=cond,
            n_units_latent=self.generator_n_units_hidden,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="none",
            generator_lr=self.lr,
            generator_residual=True,
            generator_n_iter=self.n_iter,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=self.weight_decay,
            generator_opt_betas=self.generator_opt_betas,
            generator_extra_penalties=self.generator_extra_penalties,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            discriminator_opt_betas=self.discriminator_opt_betas,
            encoder=self.encoder,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            lambda_identifiability_penalty=self.lambda_identifiability_penalty,
            encoder_max_clusters=self.encoder_max_clusters,
            dataloader_sampler=self.dataloader_sampler,
            device=self.device,
            patience=self.patience,
            patience_metric=self.patience_metric,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            adjust_inference_sampling=self.adjust_inference_sampling,
            # privacy
            dp_enabled=self.dp_enabled,
            dp_epsilon=self.dp_epsilon,
            dp_delta=self.dp_delta,
            dp_max_grad_norm=self.dp_max_grad_norm,
            dp_secure_mode=self.dp_secure_mode,
        )
        ## Check if there is a trained generator for this protection level
        path = None
        if self.protection_level is not None:
            path = self.find_models_by_protection_level(self.protection_level)
        if path is not None:
            try:
                self.load_model(path)
                self.saved = True
            except:
                self.model.fit(X.dataframe(), cond=cond)
        else:
            self.model.fit(X.dataframe(), cond=cond)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = kwargs["cond"]

        return self._safe_generate(self.model.generate, count, syn_schema, cond=cond)

    def get_dp_epsilon(self) -> float:
        return self.dp_epsilon
    
    def save_model(self):
        path = self.find_models_by_protection_level(self.protection_level)
        if path is None:
            print('save: ' + self.protection_level.name)
            # Save both the model state dict and the protection_level in a dictionary
            directory_path = os.path.join(self.cwd, self.directory)
            path = os.path.join(directory_path, self.protection_level.name)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'protection_level': self.protection_level.name
            }, path)
    
    def load_model(self, path):
        print('Using stored generator from disk: ' + self.protection_level.name)
        checkpoint = torch.load(path)
        model_state_dict = checkpoint.get('model_state_dict', None)
        self.model.load_state_dict(model_state_dict, strict=False)

plugin = AdsGANPlugin
