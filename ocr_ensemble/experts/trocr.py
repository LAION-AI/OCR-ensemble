from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from functools import partial

supported_models = []
supported_sizes = ['small', 'base', 'large']
default_size = 'small'
for size in supported_sizes:
    supported_models += [f'microsoft/trocr-{size}-stage1', 
                        f'microsoft/trocr-{size}-str', 
                        f'microsoft/trocr-{size}-printed', 
                        f'microsoft/trocr-{size}-handwritten']


def trocr_transform(image, processor):
    return processor(images=image, return_tensors="pt").pixel_values[0]

class TrOCRExpert():
    def __init__(self, model_name='microsoft/trocr-large-stage1'):
        print(model_name)
        assert model_name in supported_models, f"supported models: {supported_models}"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def get_transform(self):
        return partial(trocr_transform, processor=self.processor)
    
    def get_collate(self):
        return self.get_img_collate()

    def get_img_collate(self):
        def collate(images):
            return self.processor(images=images, return_tensors="pt").pixel_values
        return collate

    def process_batch(self, batch):
        generated_ids = self.model.generate(batch)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

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


class HandwrittenExpert(TrOCRExpert):
    def __init__(self, size=default_size):
        assert size in supported_sizes, f"supported sizes {supported_sizes}"
        super().__init__(f'microsoft/trocr-{size}-handwritten')

class PrintedExpert(TrOCRExpert):
    def __init__(self, size=default_size):
        assert size in supported_sizes, f"supported sizes {supported_sizes}"
        super().__init__(f'microsoft/trocr-{size}-printed')

class SceneExpert(TrOCRExpert):
    def __init__(self, size=default_size):
        assert size in supported_sizes, f"supported sizes {supported_sizes}"
        super().__init__(f'microsoft/trocr-{size}-str')

class Stage1Expert(TrOCRExpert):
    def __init__(self, size=default_size):
        assert size in supported_sizes, f"supported sizes {supported_sizes}"
        super().__init__(f'microsoft/trocr-{size}-stage1')


