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
    
    def get_name() -> str:
        return "Base Translator"
    
    


class DeepLTranslator(Translator):
    """The Best but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_token=None) -> None:
        super().__init__()
        self.auth_token = auth_token


    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(name="auth_token",description="DeepL Api Auth Token")]

    def translate(self, ocr_result: OcrResult):
        if self.auth_token is None or len(self.auth_token.strip()) == 0:
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
            return "Language not supported"
        
    def get_name() -> str:
        return "DeepL Translator"
        
    


class GoogleTranslateTranslator(Translator):
    """Translates using google translate"""

    def __init__(self, service_account_key_path="") -> None:
        super().__init__()

        self.key_path = service_account_key_path

        if len(self.key_path.strip()) > 0:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

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
    
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(name="service_account_key_path",description="Path to google application credentials")]
    
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
        return "Language not supported"
    
    def get_name() -> str:
        return "Helsinki NLP Japanese To English"
    
class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self,text="") -> None:
        super().__init__()
        self.to_write = text

    def translate(self, ocr_result: OcrResult):
        return self.to_write
    
    def get_name() -> str:
        return "Debug Translator"
    
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(name="text",description="What to write")]


def get_translators() -> list[Translator]:
    return [
        Translator,
        DeepLTranslator,
        GoogleTranslateTranslator,
        HelsinkiNlpJapaneseToEnglish,
        DebugTranslator
    ]
