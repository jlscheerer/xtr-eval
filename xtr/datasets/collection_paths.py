import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

BEIR_COLLECTION_PATH = config["BEIR_COLLECTION_PATH"]
LOTTE_COLLECTION_PATH = config["LOTTE_COLLECTION_PATH"]