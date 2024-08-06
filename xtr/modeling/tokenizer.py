import torch
from transformers import AutoTokenizer

from xtr.config import XTRConfig, XTRModel

class XTRTokenizer:
    """
    Wrapper around transformers.AutoTokenizer providing an interface compatible with DeepMind's implementation.
    """
    def __init__(self, config: XTRConfig):
        assert config.model == XTRModel.BASE_EN
        assert not config.is_multilingual()

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("google/xtr-base-en")

        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device

    def __call__(self, texts, maxlen):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer(texts, return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=maxlen)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        return input_ids, attention_mask