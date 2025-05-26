from newberryai import PII_Redaction

class PiiRedactor:
    def __init__(self):
        self.pii_red = PII_Redaction()

    def redact(self, text: str) -> str:
        return self.pii_red.ask(text)
