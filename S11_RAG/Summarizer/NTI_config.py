from enum import Enum

class FileConfig(Enum):
    FILE_NAME = 'Indian_Budget.pdf'
    INPUT_DATA_FOLDER_NAME = 'input'

class ModelConfig(Enum):
    CHUNK_SIZE = 10000
    CHUNK_OVERLAP = 20
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.5
