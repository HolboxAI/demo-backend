from newberryai import DDxChat

class DDxAssistant:
    def __init__(self):
        self.ddx_chat = DDxChat()

    def ask(self, question: str) -> str:
        return self.ddx_chat.ask(question)
