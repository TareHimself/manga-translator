from .c import C, D as Kwalski, F as J, example_external_method
import numpy as np
import torch


class Z:
    p = ""
    r = ""
    b = C()
    d = np.array([1])
    l = J()
    D = Kwalski()

    def __init__(self) -> None:
        example_external_method()


class A:
    p = ""
    r = Z()
    b = C()
    l = J()

    def run(self):
        pass

    def __init__(self) -> None:
        self.mask: torch.cuda.is_available
        example_external_method()
        self.run()
# ([\w\d]+)\((?:[\w\d:\s]+?)?\)|[\w\d\.]+?([\w\d]+)\((?:[\w\d:\s]+?)?\)|([\w\d]+)\.[\w\d\.]+