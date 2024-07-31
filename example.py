import tensorflow_text as tf_text

from xtr.utils.gfile import gfile_load
from xtr.utils.kaggle import kaggle_download_model
from xtr.modeling.xtr import XTRModel

model = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
tokenizer = tf_text.SentencepieceTokenizer(model=gfile_load(model), add_eos=True)

text = "This is a test"
print(tokenizer.tokenize(text))

print(kaggle_download_model(XTRModel.BASE_EN))