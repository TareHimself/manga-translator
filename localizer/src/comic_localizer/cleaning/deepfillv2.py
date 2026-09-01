# Adapted from https://github.com/nipponjo/deepfillv2-pytorch
import torch

from comic_localizer.cleaning.patched_ai_cleaner import PatchedAiCleaner
from comic_localizer.core.plugin import PluginArgument, StringPluginArgument
from comic_localizer.utils import get_default_torch_device


class DeepFillV2Cleaner(PatchedAiCleaner):
    """Cleans using Free-Form Image Inpainting with Gated Convolution https://arxiv.org/abs/1806.03589"""

    def __init__(
        self,
        model_path: str,
        inpaint_patches=True,
        patch_padding=4,
        device: torch.device = get_default_torch_device(),
        grouping_bucket_size: int = 512,
        max_group_pixels: int = 2048 * 2048 * 4,
    ) -> None:
        # set before super().__init__, which calls self.load_model()
        self.model_path = model_path
        super().__init__(
            inpaint_patches=inpaint_patches,
            patch_padding=patch_padding,
            device=device,
            grouping_bucket_size=grouping_bucket_size,
            max_group_pixels=max_group_pixels,
        )

    def load_model(self, device: torch.device) -> torch.jit.ScriptModule:
        return torch.jit.load(self.model_path, map_location=device).eval()  # type: ignore[return-value]

    @staticmethod
    def get_name() -> str:
        return "DeepFillV2"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                "model_path", "Model Path", "Path to the torchscript model"
            ),
            *PatchedAiCleaner.get_arguments(),
        ]

    @staticmethod
    def is_valid() -> bool:
        return True
