import os
from . import config
import librosa
import numpy as np


def get_info_from_filename(filename):
    tokens = filename.split('.')[0].split('-')
    emotion = int(tokens[2])-1
    emotional_intensity = int(tokens[3])-1  # 1 strong 0 normal
    # statement=int(tokens[4])
    # repetition=int(tokens[5])
    actor = int(tokens[6]) % 2  # 1 male 0 female

    return emotion, emotional_intensity, actor


def get_label_from_filename(filename):
    emotion, emotional_intensity, actor = get_info_from_filename(filename)
    return emotion


def extract_features(audio_path, sr=config.SAMPLE_RATE, offset=0, duration=3):
    try:
        y, sr_orig = librosa.load(audio_path, sr=None, offset=offset)
        if duration:
            target_duration = int(sr_orig*duration)
            if y.shape[0] < target_duration:
                y = np.pad(y, (0, target_duration-y.shape[0]), 'constant')
            else:
                y = y[:target_duration]
        if sr_orig != sr:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)

            # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=config.N_MELS,
            hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )
        mel_spec_db = librosa.power_to_db(mel_spec)  # 转为 dB 更稳定

        # MFCC及其差分（注意与 mel 参数统一）
        mfcc = librosa.feature.mfcc(
            S=mel_spec_db, sr=sr, n_mfcc=config.N_MFCC
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(
            y=y, hop_length=config.HOP_LENGTH, frame_length=config.N_FFT
        )

        # RMS Energy
        rms = librosa.feature.rms(
            y=y, hop_length=config.HOP_LENGTH, frame_length=config.N_FFT
        )

        # LLDs
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )
        flatness = librosa.feature.spectral_flatness(
            y=y, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT
        )

        # [dims, frames]
        features = np.concatenate([
            contrast,                                # 6
            centroid, bandwidth, flatness, rolloff,  # 1×4
            zcr, rms,                                # 1×2
            chroma,                                  # 12
            mel_spec_db,                             # 128
            mfcc, mfcc_delta, mfcc_delta2            # 40×3 = 120
        ], axis=0)  # shape: [dims, frames]
        
        if np.iscomplexobj(features):
            print(f"Warning: features contains complex values, taking real part.")
            features = np.real(features)

        return features.T  # shape: [frames, dims]

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def preprocess_data():
    # add one-hot gender and strength features, then save the features and labels to a file
    try:
        print(f'Processing data from {config.RAW_DATA_DIR}')
        shape = None
        for base, dirs, files in os.walk(config.RAW_DATA_DIR):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(base, file)
                    label, emotional_intensity, actor = get_info_from_filename(
                        file)
                    meta_feature = np.array(
                        [emotional_intensity, emotional_intensity ^ 1, actor, actor ^ 1])  # one-hot encoding
                    orig_feature = extract_features(file_path)
                    meta_repeated = np.tile(
                        meta_feature, (orig_feature.shape[0], 1))
                    feature = np.concatenate(
                        (meta_repeated, orig_feature), axis=1)

                    rel_path = os.path.relpath(file_path, config.RAW_DATA_DIR)
                    rel_path = os.path.splitext(rel_path)[0]+'.npz'
                    save_path = os.path.join(
                        config.PROCESSED_DATA_DIR, rel_path)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    np.savez(save_path, feature=feature, label=label)
                    print(
                        f"Processed {file_path} and saved to {save_path}\r",)

                    if not shape:
                        shape = feature.shape
                    else:
                        assert shape == feature.shape, f"Shape mismatch: expected shape={shape} but got {feature.shape}"
        print(f'Done. Feature shape:{shape}')
    except Exception as e:
        print(f"Error preprocessing data: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=config.RAW_DATA_DIR, help='Directory of raw data')
    parser.add_argument('--processed_data_dir', type=str,
                        default=config.PROCESSED_DATA_DIR, help='Directory of processed data')
    args = parser.parse_args()

    preprocess_data()
