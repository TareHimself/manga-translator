# TorchScript export of dreMaz/AnimeMangaInpainting (LaMa "big-lama" finetuned
# on manga/anime): https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript
import torch
from huggingface_hub import hf_hub_download

from comic_localizer.cleaning.patched_ai_cleaner import PatchedAiCleaner
from comic_localizer.core.plugin import PluginArgument, StringPluginArgument
from comic_localizer.utils import get_default_torch_device

_MODEL_REPO = "TareHimself/AnimeMangaInpainting-torchscript"
_MODEL_FILE = "anime_manga_lama.pt"
_MODEL_REVISION = "v1"


class LamaCleaner(PatchedAiCleaner):
    """
    Cleans using LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions https://arxiv.org/abs/2109.07161

    Downloads a torchscript export from the Hub by default (pinned to a release
    tag); set `model_path` to load a local file instead.
    """

    def __init__(
        self,
        model_path: str = "",
        repo: str = _MODEL_REPO,
        filename: str = _MODEL_FILE,
        revision: str = _MODEL_REVISION,
        inpaint_patches=True,
        patch_padding=4,
        device: torch.device = get_default_torch_device(),
        grouping_bucket_size: int = 512,
        max_group_pixels: int = 2048 * 2048 * 4,
    ) -> None:
        # set before super().__init__, which calls self.load_model()
        self.model_path = model_path
        self.repo = repo
        self.filename = filename
        self.revision = revision
        super().__init__(
            inpaint_patches=inpaint_patches,
            patch_padding=patch_padding,
            device=device,
            grouping_bucket_size=grouping_bucket_size,
            max_group_pixels=max_group_pixels,
        )

    def load_model(self, device: torch.device) -> torch.jit.ScriptModule:
        path = self.model_path or hf_hub_download(
            self.repo, self.filename, revision=self.revision or None
        )
        return torch.jit.load(path, map_location=device).eval()  # type: ignore[return-value]

    @staticmethod
    def get_name() -> str:
        return "Lama"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                "model_path",
                "Model Path",
                "Local torchscript file. Leave blank to download from the Hub.",
            ),
            StringPluginArgument(
                "repo",
                "Model Repo",
                "HuggingFace repo id (used when Model Path is blank)",
                _MODEL_REPO,
            ),
            StringPluginArgument(
                "filename", "Model File", "File within the repo", _MODEL_FILE
            ),
            StringPluginArgument(
                "revision",
                "Revision",
                "Tag / branch / commit (blank = latest)",
                _MODEL_REVISION,
            ),
            *PatchedAiCleaner.get_arguments(),
        ]

    @staticmethod
    def is_valid() -> bool:
        return True
