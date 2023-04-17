from .proposers import PaddleOCRProposalGenerator
from .experts import Ensemble, HandwrittenExpert, PaddleOCRExpert

class OCR():
    def __init__(self):
        self.proposer = PaddleOCRProposalGenerator()
        self.ensemble = Ensemble({"contains printed text": PaddleOCRExpert(),
                                  "contains handwritten text": HandwrittenExpert()})
    
    def __call__(self, image):
        # expects [height, width, channels] 0..255 uint8
        crops, bboxes = self.proposer(image)
        txts = self.ensemble.process_list(crops)
        return bboxes, txts
