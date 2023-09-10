from translator.utils import get_languages
from translator.core.plugin import Translator, TranslatorResult, OcrResult, PluginSelectArgument, PluginSelectArgumentOption, PluginTextArgument, PluginArgument


class OpenAiTranslator(Translator):
    """Uses an Open Ai Model for translation"""

    MODELS = [
        ("GPT 3.5 Turbo","gpt-3.5-turbo"),
        ("GPT 4","gpt-4"),
        ("GPT 4 0613","gpt-4-0613"),
        ("GPT 3.5 Turbo 16K","gpt-3.5-turbo-16k"),
        ("GPT 3.5 Turbo 0613","gpt-3.5-turbo-0613")
    ]
    
    def __init__(self, api_key="",target_lang = "en",model=MODELS[0][1],temp="0.2") -> None:
        super().__init__()
        import openai
        openai.api_key = api_key
        self.openai = openai
        self.target_lang = target_lang
        self.model = model
        self.temp = float(temp)
        

    def translate(self, ocr_result: OcrResult):
        if len(ocr_result.text.strip()) == 0:
            return TranslatorResult(lang_code=self.target_lang)
        
        message = f"{ocr_result.language.upper()} to {self.target_lang.upper()}\n{ocr_result.text}"
        
        result = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role":"user","content": "EN to JA\nHello"},
                {"role": "assistant", "content":"こんにちは"},
                {"role":"user","content": message }
                ],
            
        )
        return TranslatorResult(result['choices'][0].message['content'].strip(),self.target_lang)

    @staticmethod
    def get_name() -> str:
        return "GPT Translator"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        languages = get_languages()
        languages.sort(key= lambda a: a[0].lower())
        options = list(map(lambda a : PluginSelectArgumentOption(a[0],a[1]),languages))

        return [PluginTextArgument(id="api_key", name="API Key", description="Your api Key"),
                PluginSelectArgument(id="target_lang",name="Target Language",description="The language to translate to",options=options,default="en"),
                PluginSelectArgument(id="model",name="Model",description="The model to use",options=list(map(lambda a: PluginSelectArgumentOption(a[0],a[1]),OpenAiTranslator.MODELS)),default=OpenAiTranslator.MODELS[0][1])]
