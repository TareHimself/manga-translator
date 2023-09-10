import requests
import traceback
from requests.utils import requote_uri
from translator.core.plugin import Translator,OcrResult,TranslatorResult,PluginArgument,PluginSelectArgument,PluginSelectArgumentOption,PluginTextArgument

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
        
        if len(ocr_result.text.strip()) == 0:
            return TranslatorResult()

        if ocr_result.language == "ja":
            try:
                data = [("target_lang", "EN-US"), ("source_lang", "JA"), ("text", ocr_result.text)]
                uri = f"https://api-free.deepl.com/v2/translate?{'&'.join([f'{data[i][0]}={data[i][1]}' for i in range(len(data))])}"
                uri = requote_uri(uri)

                return TranslatorResult(requests.post(
                    uri,
                    headers={
                        "Authorization": f"DeepL-Auth-Key {self.auth_token}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                ).json()["translations"][0]["text"])
            except:
                traceback.print_exc()
                return TranslatorResult("Failed To Get Translation")
        else:
            return TranslatorResult("Language not supported")

    @staticmethod
    def get_name() -> str:
        return "DeepL Translator"