import numpy
import requests
import traceback
import os
from transformers import pipeline
from requests.utils import requote_uri
from translator.core.ocr import BaseOcr, OcrResult
from translator.core.plugin import BasePlugin, PluginArgument, PluginTextArgument


class Translator(BasePlugin):
    """Base Class for all Translator classes"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, ocr: BaseOcr, text: numpy.ndarray) -> str:
        return self.translate(self.apply_ocr(ocr, text))

    def apply_ocr(self, ocr: BaseOcr, text: numpy.ndarray) -> OcrResult:
        return ocr(text)

    def translate(self, ocr_result: OcrResult) -> str:
        return ocr_result.text

    @staticmethod
    def get_name() -> str:
        return "Base Translator"


class DeepLTranslator(Translator):
    """The Best but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_token=None) -> None:
        super().__init__()
        self.auth_token = auth_token

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(id="auth_token", name="Auth Token", description="DeepL Api Auth Token")]

    def translate(self, ocr_result: OcrResult):
        if self.auth_token is None or len(self.auth_token.strip()) == 0:
            return "Need DeepL Auth"

        if ocr_result.language == "ja":
            try:
                data = [("target_lang", "EN-US"), ("source_lang", "JA"), ("text", ocr_result.text)]
                uri = f"https://api-free.deepl.com/v2/translate?{'&'.join([f'{data[i][0]}={data[i][1]}' for i in range(len(data))])}"
                uri = requote_uri(uri)

                return requests.post(
                    uri,
                    headers={
                        "Authorization": f"DeepL-Auth-Key {self.auth_token}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                ).json()["translations"][0]["text"]
            except:
                traceback.print_exc()
                return "Failed To Get Translation"
        else:
            return "Language not supported"

    @staticmethod
    def get_name() -> str:
        return "DeepL Translator"


class GoogleTranslateTranslator(Translator):
    """Translates using Google Translate"""

    def __init__(self, key_path="") -> None:
        super().__init__()

        self.key_path = key_path

        if len(self.key_path.strip()) > 0:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

            from google.cloud import translate_v2 as translate

            self.trans = translate.Client()
        else:
            self.trans = None

    def translate(self, ocr_result: OcrResult):
        if self.trans is None:
            return "Invalid Key Path"

        return self.trans.translate(
            ocr_result.text, source_language=ocr_result.language, target_language="en"
        )["translatedText"]

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(id="key_path", name="Service Account Key Path",
                                   description="Path to google application credentials json")]

    @staticmethod
    def get_name() -> str:
        return "Google Cloud Translate"


class HuggingFace(Translator):
    """Translates using hugging face models"""

    def __init__(self, model_url: str = "Helsinki-NLP/opus-mt-ja-en") -> None:
        super().__init__()
        self.pipeline = pipeline("translation", model=model_url)

    def translate(self, ocr_result: OcrResult):
        return self.pipeline(ocr_result.text)[0]["translation_text"]

    @staticmethod
    def get_name() -> str:
        return "Hugging Face"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            PluginTextArgument(id="model_url", name="Model", description="The Hugging Face translation model to use",
                               default="staka/fugumt-ja-en")]


class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self, text="") -> None:
        super().__init__()
        self.to_write = text

    def translate(self, ocr_result: OcrResult):
        return self.to_write

    @staticmethod
    def get_name() -> str:
        return "Debug Translator"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(id="text", name="Debug Text", description="What to write")]


class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self, api_key="") -> None:
        super().__init__()
        import openai
        openai.api_key = api_key
        

    def translate(self, ocr_result: OcrResult):
        return self.to_write

    @staticmethod
    def get_name() -> str:
        return "GPT Translator"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(id="api_key", name="Open Ai Api Key", description="Your api Key")]


def get_translators() -> list[Translator]:
    return list(filter(lambda a: a.is_valid(), [
        Translator,
        DeepLTranslator,
        GoogleTranslateTranslator,
        HuggingFace,
        DebugTranslator
    ]))
