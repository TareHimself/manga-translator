import numpy
import requests
import traceback
import os
from transformers import pipeline
from requests.utils import requote_uri
from translator.utils import cv2_to_pil
from translator.ocr import BaseOcr, OcrResult
from translator.plugin import BasePlugin, PluginArgument

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
    
    def get_name() -> str:
        return "Base Translator"
    
    


class DeepLTranslator(Translator):
    """The Best but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_token=None) -> None:
        super().__init__()
        self.auth_token = auth_token


    def get_arguments() -> list[PluginArgument]:
        return [PluginArgument(name="auth_token",description="DeepL Api Auth Token",required=True)]

    def translate(self, ocr_result: OcrResult):
        if self.auth_token is None:
            return "Need DeepL Auth"

        if ocr_result.language == "ja":
            try:
                data = [("target_lang", "EN-US"), ("source_lang", "JA")]
                data.append(("text", ocr_result.text))
                uri = f"https://api-free.deepl.com/v2/translate?{'&'.join([f'{data[i][0]}={data[i][1]}' for i in range(len(data))])}"
                uri = requote_uri(uri)
                
                return requests.post(
                    uri,
                    headers={
                        "Authorization": f"DeepL-Auth-Key {self.auth_token}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                ).json()["translations"][0]["text"]
            except Exception as e:
                traceback.print_exc()
                return "Failed To Get Translation"
        else:
            return ocr_result.text
        
    def get_name() -> str:
        return "DeepL Translator"
        
    


class GoogleTranslateTranslator(Translator):
    """Translates using google translate"""

    def __init__(self, service_account_key_path="") -> None:
        super().__init__()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
        from google.cloud import translate_v2 as translate

        self.trans = translate.Client()

    def translate(self, ocr_result: OcrResult):
        return self.trans.translate(
            ocr_result.text, source_language=ocr_result.language, target_language="en"
        )["translatedText"]
    
    def get_arguments() -> list[PluginArgument]:
        return [PluginArgument(name="service_account_key_path",description="Path to google application credentials",required=True)]
    
    def get_name() -> str:
        return "Google Cloud Translate"


class HelsinkiNlpJapaneseToEnglish(Translator):
    """Translates using this model on hugging face https://huggingface.co/Helsinki-NLP/opus-mt-ja-en"""

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")

    def translate(self, ocr_result: OcrResult):
        if ocr_result.language == "ja":
            return self.pipeline(ocr_result.text)[0]["translation_text"]

        return super().translate(ocr_result)
    
    def get_name() -> str:
        return "Helsinki NLP Japanese To English"


def get_translators() -> list[Translator]:
    return [
        Translator,
        DeepLTranslator,
        GoogleTranslateTranslator,
        HelsinkiNlpJapaneseToEnglish,
    ]
