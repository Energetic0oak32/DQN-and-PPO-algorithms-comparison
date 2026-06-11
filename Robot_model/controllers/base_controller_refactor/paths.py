from config import ALGORITHM, MODEL_VERSION


def model_path():
    return f"{ALGORITHM.lower()}_pioneer_{MODEL_VERSION}_model"


def replay_buffer_path():
    return f"{ALGORITHM.lower()}_pioneer_{MODEL_VERSION}_replay_buffer.pkl"
