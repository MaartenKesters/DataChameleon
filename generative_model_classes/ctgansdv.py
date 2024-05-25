"""
References: 
- "Modeling Tabular data using Conditional GAN", Xu, Lei  et al. (https://arxiv.org/abs/1907.00503)
- "The synthetic data vault", Patki, Neha et al. (https://ieeexplore.ieee.org/abstract/document/7796926)
- "Synthcity: facilitating innovative use cases of synthetic data in different data modalities", Zhaozhi, Qian et al. (https://arxiv.org/abs/2301.07573)
"""

# stdlib
from pathlib import Path
from typing import Any, List

# Necessary packages
from pydantic import validate_arguments
import torch
import os

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE

# SDV
from ctgan import CTGAN

from generative_model_classes.plugin import Plugin
from protectionLevel import ProtectionLevel


class CTGANSDV(Plugin):
    """
    Conditional Table GAN Synthesizer.
    Args:
        protection_level (ProtectionLevel):
            Protection of level of the generator.
        discrete_columns (list of strings):
            List of discrete columns to be used to generate the Conditional
            Vector. This list should contain the column names.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
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
        epochs=10,
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
        self.directory = os.path.join("generators", "CTGANSDV")

    @staticmethod
    def name() -> str:
        return "ctgansdv"

    @staticmethod
    def type() -> str:
        return "privacy"

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "CTGANSDV":
        self.model = CTGAN(**self._model_kwargs)
        ## Check if there is a trained model for this protection level
        path = None
        if self.protection_level is not None:
            path = self.find_models_by_protection_level(self.protection_level)
        if path is not None:
            try:
                ## Load the trained model
                self.load_model(path)
                self.saved = True
            except:
                self.model.fit(X.dataframe(), discrete_columns=self.discrete_columns)
        else:
            self.model.fit(X.dataframe(), discrete_columns=self.discrete_columns)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        return self._safe_generate(self.model.sample, count, syn_schema)
    
    def save_model(self):
        path = self.find_models_by_protection_level(self.protection_level)
        if path is None:
            print('save: ' + self.protection_level.name)
            # Save both the model state dict and the protection_level in a dictionary
            directory_path = os.path.join(self.cwd, self.directory)
            path = os.path.join(directory_path, self.protection_level.name)
            torch.save({
                'model_state_dict': self.model.__getstate__(),
                'protection_level': self.protection_level.name
            }, path)
    
    def load_model(self, path):
        print('Using stored generator from disk: ' + self.protection_level.name)
        checkpoint = torch.load(path)
        model_state_dict = checkpoint.get('model_state_dict', None)
        self.model.__setstate__(model_state_dict)

plugin = CTGANSDV