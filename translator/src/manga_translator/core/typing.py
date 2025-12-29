import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray
from typing import Annotated
# Vector3: TypeAlias = NDArray[np.float32]  # dtype only
Vector2: TypeAlias = Annotated[NDArray[np.float32], (2)]
Vector2i: TypeAlias = Annotated[NDArray[np.int32], (2)]
Vector4i: TypeAlias = Annotated[NDArray[np.int32], (4)]
Array2x2: TypeAlias = Annotated[NDArray[np.int32], (2, 2)]