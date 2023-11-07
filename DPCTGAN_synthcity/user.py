# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.schema import Schema

from privacyLevel import PrivacyLevels

class User():
    """
    Class for the users of the data chameleon.

    Constructor Args:
        
    """

    def __init__(
        self,
        name: str,
        data_owner: bool = False,
        privacy_level: PrivacyLevels = PrivacyLevels.SECRET
        ):
        self.name = name
        self.data_owner = data_owner
        if (data_owner):
            self.privacy_level = PrivacyLevels.LOW
        else:
            self.privacy_level = privacy_level
        # keep track if the user already requested synthetic data. If not, possible to use previously generated data. If yes, need to create new synthetic data.
        self.data_requested = False

    def get_name(self):
        return self.name
    
    def get_data_owner(self):
        return self.data_owner
    
    def get_privacy_level(self):
        return self.privacy_level
    
    def set_privacy_level(self, level):
        if self.privacy_level != level:
            self.privacy_level = level
            self.data_requested = False

    def get_data_requested(self):
        return self.data_requested
    
    def set_data_requested(self, value):
        self.data_requested = value