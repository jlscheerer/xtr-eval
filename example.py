from xtr.config import XTRModel, XTRConfig, XTRBruteForceIndexConfig

config = XTRConfig(model=XTRModel.BASE_EN, index_config=XTRBruteForceIndexConfig())
print(config.index_type)