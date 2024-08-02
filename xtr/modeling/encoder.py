import torch

from transformers import AutoModel, logging
from huggingface_hub import hf_hub_download

from xtr.config import XTRConfig, XTRModel
from xtr.modeling.tokenizer import XTRTokenizer

# Source: https://huggingface.co/google/xtr-base-en/resolve/main/2_Dense/config.json
# Activation function is torch.nn.modules.linear.Identity
class XTRLinear(torch.nn.Module):
    def __init__(self, in_features=768, out_features=128, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x):
        return self.linear(x)

class XTREncoder(torch.nn.Module):
    """
    Wrapper around transformers.AutoModel providing an interface compatible with DeepMind's implementation.
    In particular, this module incorporates XTR's final linear layer.
    """
    def __init__(self, config: XTRConfig):
        super().__init__()

        assert config.model == XTRModel.BASE_EN

        # NOTE Warning about unitialized decoder weights is to be expected.
        #      We only make use of the encoder anyways.
        logging.set_verbosity_error()
        model = AutoModel.from_pretrained("google/xtr-base-en", use_safetensors=True)

        self.encoder = model.encoder

        # Source: https://huggingface.co/google/xtr-base-en/
        to_dense_path = hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")
        self.linear = XTRLinear()
        self.linear.load_state_dict(torch.load(to_dense_path, weights_only=True))

        logging.set_verbosity_warning()

        self.tokenizer = XTRTokenizer(config)

    def forward(self, texts):
        with torch.inference_mode():
            input_ids, attention_mask = self.tokenizer(texts)
            Q = self.encoder(input_ids, attention_mask).last_hidden_state
            Q = self.linear(Q)
            mask = (input_ids != 0).unsqueeze(2).float()
            Q = Q * mask
            Q = torch.nn.functional.normalize(Q, dim=2)
            return {
                "encodings": Q,
                "mask": mask,
            }