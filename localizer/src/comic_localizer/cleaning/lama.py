# TorchScript export of dreMaz/AnimeMangaInpainting (LaMa "big-lama" finetuned
# on manga/anime): https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript
from comic_localizer.cleaning.patched_ai_cleaner import PatchedAiCleaner


class LamaCleaner(PatchedAiCleaner):
    """
    Cleans using LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions https://arxiv.org/abs/2109.07161
    """

    DEFAULT_MODEL_REPO = "TareHimself/AnimeMangaInpainting-torchscript"
    DEFAULT_MODEL_FILE = "anime_manga_lama.pt"

    @staticmethod
    def get_name() -> str:
        return "Lama"

    @staticmethod
    def is_valid() -> bool:
        return True
