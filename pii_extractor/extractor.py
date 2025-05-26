from newberryai import PII_extraction

class PiiExtractor:
    def __init__(self):
        self.pii_extract = PII_extraction()

    def extract(self, text: str) -> dict:
        return self.pii_extract.ask(text)
