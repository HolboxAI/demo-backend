from newberryai import MedicalCoder

class MedicalCodingAssistant:
    def __init__(self):
        self.coder = MedicalCoder()

    def extract_codes(self, file_path: str) -> dict:
        """
        Extract ICD-10 and CPT codes from a medical document.
        
        Args:
            file_path (str): Path to the medical document (PDF/Image/Text).
        
        Returns:
            dict: Extracted ICD-10 and CPT codes along with descriptions.
        """
        return self.coder.ask(file_path)