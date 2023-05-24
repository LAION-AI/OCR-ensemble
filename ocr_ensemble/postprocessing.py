import openai
from transformers import AutoTokenizer, RobertaForMaskedLM, RobertaPreLayerNormForMaskedLM, XLMRobertaForMaskedLM
import torch
from weighted_levenshtein import lev


class RobertaPostprocessor:
    def __init__(self, debug=False, size="base"):
        self.tokenizer = AutoTokenizer.from_pretrained(f"roberta-{size}")
        self.model = RobertaForMaskedLM.from_pretrained(f"roberta-{size}")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def __call__(self, text, filter=False):
        if isinstance(text, str):
            input_text = [text]
        else:
            input_text = text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        out = self.tokenizer.batch_decode(logits.argmax(dim=2))
        out = [out.replace('<s>', '').replace('</s>', '').strip() for out in out]
        if filter:
            for idx, o in enumerate(out):
                if lev(o.encode("ascii", "ignore").decode(), input_text[idx].encode("ascii", "ignore").decode()) > 5:
                    out[idx] = input_text[idx]
        if isinstance(text, str):
            return out[0]
        return out

class RobertaPreLNPostprocessor:
    def __init__(self, debug=False):
        self.model = RobertaPreLayerNormForMaskedLM.from_pretrained("andreasmadsen/efficient_mlm_m0.40")
        self.tokenizer = AutoTokenizer.from_pretrained("andreasmadsen/efficient_mlm_m0.40")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def __call__(self, text, filter=False):
        if isinstance(text, str):
            input_text = [text]
        else:
            input_text = text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        out = self.tokenizer.batch_decode(logits.argmax(dim=2))
        out = [out.replace('<s>', '').replace('</s>', '').strip() for out in out]
        if filter:
            for idx, o in enumerate(out):
                if lev(o.encode("ascii", "ignore").decode(), input_text[idx].encode("ascii", "ignore").decode()) > 5:
                    out[idx] = input_text[idx]
        if isinstance(text, str):
            return out[0]
        return out
    
class XLMRobertaPostprocessor:
    def __init__(self, debug=False, size="base"):
        self.tokenizer = AutoTokenizer.from_pretrained(f"xlm-roberta-{size}")
        self.model = XLMRobertaForMaskedLM.from_pretrained(f"xlm-roberta-{size}")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def __call__(self, text, filter=False):
        if isinstance(text, str):
            input_text = [text]
        else:
            input_text = text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        out = self.tokenizer.batch_decode(logits.argmax(dim=2))
        out = [out.replace('<s>', '').replace('</s>', '').strip() for out in out]
        if filter:
            for idx, o in enumerate(out):
                if lev(o.encode("ascii", "ignore").decode(), input_text[idx].encode("ascii", "ignore").decode()) > 5:
                    out[idx] = input_text[idx]
        if isinstance(text, str):
            return out[0]
        return out


def postprocess(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a masterful text postprocessor."},
            {"role": "user", "content": "I need you to simulate a text postprocessor. When I write postprocess('texttisbetter') you respond with a 'text is better' replacing the input to postprocess with the most plausible text. No explanations, no small talk, no questions, no excuses, only the result. If the input is too hard, just return the input. postprocess('BECOMINGMORE')"},
            {"role": "assistant", "content": "BECOMING MORE"},
            {"role": "user", "content": "postprocess('%s')"%text}
        ]
    )
    return response["choices"][0]["message"]["content"]