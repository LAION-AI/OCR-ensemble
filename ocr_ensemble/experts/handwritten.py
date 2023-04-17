from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

class HandwrittenExpert():
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')


    def process_list(self, images):
        # list of h, w, c images
        # b, h, w, c images of text
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def __call__(self, image):
        # h, w, c images of text
        return self.process_list(image[np.newaxis])[0]
        
