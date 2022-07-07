from easynmt import EasyNMT
from regex import B


class Translator:
    def __init__(self):

        self.model = EasyNMT("opus-mt")
        self.source_lang = "mul"
        self.target_lang = "en"
        self.beam_size = 5
        self.batch_size = 1

        self.lang = "None"
        self.sent = "None"

    def identification(self, sentence):
        self.lang = self.model.language_detection(sentence)

    def translation(self, sentence):
        self.identification(sentence)
        self.sent = self.model.translate(
            sentence,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
        )
