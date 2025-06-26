import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional 
from pathlib import Path

@dataclass
class ProcessingConfig:
    
    """Configuration of the image processing parameters using type hinting"""
    
    XScale: int = 200 
    Max_points: int = 200
    Scale: np.ndarray = field(default_factory = lambda: np.arange(10, 31))
    Weighted_sampling: int = 1
    Workspace_dir: Optional[Path] = None # The use of Optional here is a special case
                                         # to indicate that this field can be either a
                                         # Path object or None (it's the equivalent of
                                         # Union(Path, None)). 
    Plot: bool = False