from typing import Any
from ultralytics import YOLO
import numpy as np
from manga_ocr import MangaOcr
import requests
import traceback
from requests.utils import requote_uri
from utils import cv2_to_pil


class Translator:
    def __init__(self) -> None:
        pass

    def __call__(self, text: np.ndarray) -> Any:
        return self.translate(text)

    def translate(self, text: np.ndarray):
        return "Sample"


class DeepLTranslator(Translator):
    def __init__(self, auth_token=None) -> None:
        super().__init__()
        self.auth_token = auth_token
        self.ocr = MangaOcr()

    def translate(self, text: np.ndarray):
        if self.auth_token is None:
            return "Need DeepL Auth"

        text = self.ocr(cv2_to_pil(text))
        print(text)
        try:
            data = [("target_lang", "EN-US"), ("source_lang", "JA")]
            data.append(("text", text))
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


class GoogleTranslateTranslator(Translator):
    def __init__(self) -> None:
        super().__init__()

    def translate(self, text: np.ndarray):
        return super().translate(text)
