# handwritten.py

from newberryai import Handwrite2Text

def extract_handwritten_text(image_path: str) -> str:
    handwriter = Handwrite2Text()
    return handwriter.extract_text(image_path)
