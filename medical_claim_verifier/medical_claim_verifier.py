from newberryai import MedicalClaimVerifier

class MedicalClaimVerifierAssistant:
    def __init__(self):
        self.verifier = MedicalClaimVerifier()

    def verify_claim_from_document(self, file_path: str, insurance_provider: str = None) -> dict:
        """
        Verify a medical claim from a document (PDF/Image/Text)
        
        Args:
            file_path (str): Path to the medical claim document
            insurance_provider (str, optional): Name of the insurance provider
            
        Returns:
            dict: Verification results including approval prediction, risk factors, and recommendations
        """
        if insurance_provider:
            return self.verifier.verify_claim_from_document(file_path, insurance_provider)
        else:
            return self.verifier.verify_claim_from_document(file_path)
