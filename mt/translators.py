import numpy
import requests
import traceback
from transformers import pipeline
from requests.utils import requote_uri
from mt.utils import cv2_to_pil
from mt.ocr import BaseOcr, OcrResult


class Translator:
    def __init__(self) -> None:
        pass

    def __call__(self, ocr: BaseOcr, text: numpy.ndarray) -> str:
        return self.translate(self.apply_ocr(ocr, text))

    def apply_ocr(self, ocr: BaseOcr, text: numpy.ndarray) -> OcrResult:
        return ocr(cv2_to_pil(text))

    def translate(self, ocr_result: OcrResult) -> str:
        return ocr_result.text


class DeepLTranslator(Translator):
    def __init__(self, auth_token=None) -> None:
        super().__init__()
        self.auth_token = auth_token

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


class GoogleTranslateTranslator(Translator):
    def __init__(self) -> None:
        super().__init__()

    def translate(self, ocr_result: OcrResult):
        return super().translate(ocr_result)


class HelsinkiNlpJapaneseToEnglish(Translator):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")

    def translate(self, ocr_result: OcrResult):
        if ocr_result.language == "ja":
            return self.pipeline(ocr_result.text)[0]["translation_text"]

        return super().translate(ocr_result)
