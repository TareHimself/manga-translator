import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray
from typing import Annotated
# Vector3: TypeAlias = NDArray[np.float32]  # dtype only
Vector2: TypeAlias = np.ndarray[(2),np.float32]
Vector2i: TypeAlias =  np.ndarray[(2),np.int32]
Vector3u8: TypeAlias = np.ndarray[(3),np.uint8]
Vector4i: TypeAlias = np.ndarray[(4),np.int32]