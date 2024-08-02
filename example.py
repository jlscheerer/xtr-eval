from xtr.config import XTRConfig, XTRModel, XTRBruteForceIndexConfig
from xtr.modeling.xtr import XTR

config = XTRConfig(model=XTRModel.BASE_EN, index_config=XTRBruteForceIndexConfig())
xtr = XTR(config=config)