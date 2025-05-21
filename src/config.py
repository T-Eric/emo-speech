EMOTION = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

EMOTIONAL_INTENSITY = {
    0: "normal", 1: "strong"
}

# Librosa默认采样率有时是22050Hz，RAVDESS原始是48000Hz，需要统一
RAW_DATA_DIR = "data/dataset/"
PROCESSED_DATA_DIR = "data/processed_data/"
SAMPLE_RATE = 22050
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_PAD_LEN_MFCC = 174
MAX_PAD_LEN_MEL = 174