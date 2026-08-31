# TorchScript export of dreMaz/AnimeMangaInpainting (LaMa "big-lama" finetuned
# on manga/anime): https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript
import torch
from huggingface_hub import hf_hub_download

from comic_localizer.cleaning.patched_ai_cleaner import PatchedAiCleaner

_MODEL_REPO = "TareHimself/AnimeMangaInpainting-torchscript"
_MODEL_FILE = "anime_manga_lama.pt"
_MODEL_REVISION = "v1"


class LamaCleaner(PatchedAiCleaner):
    """
    Cleans using LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions https://arxiv.org/abs/2109.07161

    The model is pulled from the Hub, so this cleaner needs no configuration.
    """

    def load_model(self, device: torch.device) -> torch.jit.ScriptModule:
        path = hf_hub_download(_MODEL_REPO, _MODEL_FILE, revision=_MODEL_REVISION)
        return torch.jit.load(path, map_location=device).eval()  # type: ignore[return-value]

    @staticmethod
    def get_name() -> str:
        return "Lama"

    @staticmethod
    def is_valid() -> bool:
        return True
