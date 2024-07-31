import kagglehub

from xtr.modeling.xtr import XTRModel

XTR_MODEL_TO_HUB = {
    XTRModel.BASE_EN: "deepmind/xtr/tensorFlow2/base-en",
    XTRModel.BASE_EN_TPU: "deepmind/xtr/tensorFlow2/base-en-tpu",
    XTRModel.BASE_MULTILINGUAL: "deepmind/xtr/tensorFlow2/base-multilingual",
    XTRModel.BASE_MULTILINGUAL_TPU: "deepmind/xtr/tensorFlow2/base-multilingual-tpu",
    XTRModel.XXL_EN: "deepmind/xtr/tensorFlow2/xxl-en",
    XTRModel.XXL_EN_TPU: "deepmind/xtr/tensorFlow2/xxl-en-tpu",
    XTRModel.XXL_MULTILINGUAL: "deepmind/xtr/tensorFlow2/xxl-multilingual",
    XTRModel.XXL_MULTILINGUAL_TPU: "deepmind/xtr/tensorFlow2/xl-multilingual-tpu",
}

def kaggle_download_model(path: str):
    return kagglehub.model_download(XTR_MODEL_TO_HUB[path])