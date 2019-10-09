import librosa
import numpy as np
import glob
import os
import mirdata


def compute_confidence(rms_lead, rms_acc):
    conf = rms_lead / (max(rms_lead) + rms_acc)
    return conf


def pitch_confidence(audio_path, sr=44100, window_len=2048, hop_len=128):
    y, sr = librosa.load(audio_path, sr=sr)
    S, phase = librosa.magphase(librosa.stft(y, hop_length=hop_len, n_fft=window_len))
    rms = librosa.feature.rms(S=S).flatten()
    # There is no accompaniment
    acc_rms = np.zeros_like(rms)
    conf = compute_confidence(rms, acc_rms)
    times = librosa.frames_to_time(np.arange(len(conf)), sr=sr, hop_length=hop_len)
    return times, conf


def audio_to_rms(audio, sr=44100, window_len=2048, hop_len=128):
    S, phase = librosa.magphase(
        librosa.stft(audio, hop_length=hop_len, n_fft=window_len)
    )
    rms = librosa.feature.rms(S=S).flatten()
    return rms


def ikala_confidence(audio_path, sr=44100, window_len=2048, hop_len=128):
    audio, sr = librosa.load(audio_path, sr=sr, mono=False)
    vocal_channel = audio[1, :]
    rms_vocal = audio_to_rms(vocal_channel, window_len=window_len, hop_len=hop_len)

    instrumental_channel = audio[0, :]
    rms_instrumental = audio_to_rms(
        instrumental_channel, window_len=window_len, hop_len=hop_len
    )
    conf = compute_confidence(rms_vocal, rms_instrumental)
    times = librosa.frames_to_time(np.arange(len(conf)), sr=sr, hop_length=hop_len)
    return times, conf


def save_confidence(times, conf, filename, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    f0_file = get_output_path(filename, out_folder, ".conf.csv")
    f0_data = np.vstack([times, conf]).transpose()
    np.savetxt(f0_file, f0_data, fmt=["%.5f", "%.6f"], delimiter=",")
    print("Writing: " + f0_file)


def get_audio_track_paths(folder):
    audio_files = glob.glob(os.path.join(folder, "*.wav"))
    return audio_files


def get_audio_track_paths_separated_ikala(folder):
    audio_files_verse = glob.glob(os.path.join(folder, "*verse.wav"))
    audio_files_chorus = glob.glob(os.path.join(folder, "*chorus.wav"))
    all_files = audio_files_verse + audio_files_chorus
    sep = {}
    for file in all_files:
        sep[file] = {}
        acc = file[0:-4] + "_acc_VUIMM.wav"
        lead = file[0:-4] + "_lead_VUIMM.wav"
        sep[file]["acc"] = acc
        sep[file]["lead"] = lead
    return sep


def get_audio_track_paths_separated(folder):

    audio_files_verse = glob.glob(os.path.join(folder, "*verse.wav"))
    audio_files_chorus = glob.glob(os.path.join(folder, "*chorus.wav"))
    all_files = audio_files_verse + audio_files_chorus
    sep = {}
    for file in all_files:
        sep[file] = {}
        acc = file[0:-4] + "_acc_VUIMM.wav"
        lead = file[0:-4] + "_lead_VUIMM.wav"
        sep[file]["acc"] = acc
        sep[file]["lead"] = lead
    return sep


def get_output_path(wav_path, output_dir, suffix=".mel"):
    """
    return the output path for a given wav file
    """
    new_path = wav_path + suffix
    path = os.path.join(output_dir, os.path.basename(new_path))
    return path


def confidence_separated(separated_audio_paths, sr=44100, window_len=2048, hop_len=128):
    vocal_channel, sr = librosa.load(separated_audio_paths["lead"], sr=sr)
    instrumental_channel, sr = librosa.load(separated_audio_paths["acc"], sr=sr)

    rms_vocal = audio_to_rms(vocal_channel, window_len=window_len, hop_len=hop_len)
    rms_instrumental = audio_to_rms(
        instrumental_channel, window_len=window_len, hop_len=hop_len
    )

    conf = compute_confidence(rms_vocal, rms_instrumental)
    times = librosa.frames_to_time(np.arange(len(conf)), sr=sr, hop_length=hop_len)
    return times, conf


def get_audio_track_paths_separated_orchset():
    orchset_data = mirdata.orchset.load()
    sep = {}
    for key, track in orchset_data.items():
        file = track.audio_path_mono
        sep[file] = {}
        acc = file[0:-4] + "_acc_VUIMM.wav"
        lead = file[0:-4] + "_lead_VUIMM.wav"
        sep[file]["acc"] = acc
        sep[file]["lead"] = lead
    return sep


def compute_separated_confidence(sep_file_data, window_len=2048, hop_len=128):
    for key, paths in sep_file_data.items():
        print(paths["lead"])
        print(paths["acc"])
        times, conf = confidence_separated(
            paths, window_len=window_len, hop_len=hop_len
        )
        out_folder = os.path.join(os.path.dirname(paths["lead"]), "conf")
        save_confidence(times, conf, paths["lead"], out_folder)


def main():
    window_len = 2048 * 2
    hop_len = 128 * 2

    # ikala source separated
    sep_ikala_paths = get_audio_track_paths_separated_ikala(
        "/home/juanjoseb/mir_datasets/iKala/mono"
    )
    compute_separated_confidence(
        sep_ikala_paths, window_len=window_len, hop_len=hop_len
    )

    # medleydb pitch
    pitch_paths = get_audio_track_paths(
        "/home/juanjoseb/mir_datasets/MedleyDB-Pitch/audio"
    )
    for audio_file in pitch_paths:
        times, conf = pitch_confidence(
            audio_file, window_len=window_len, hop_len=hop_len
        )
        out_folder = os.path.join(os.path.dirname(audio_file), "conf")
        save_confidence(times, conf, audio_file, out_folder)

    # ikala pitch
    ikala_paths = get_audio_track_paths("/home/juanjoseb/mir_datasets/iKala/Wavfile/")
    for audio_file in ikala_paths:
        times, conf = ikala_confidence(
            audio_file, window_len=window_len, hop_len=hop_len
        )
        out_folder = os.path.join(os.path.dirname(audio_file), "conf")
        save_confidence(times, conf, audio_file, out_folder)

    # orchset source separated
    sep_orchset_paths = get_audio_track_paths_separated_orchset()
    compute_separated_confidence(
        sep_orchset_paths, window_len=window_len, hop_len=hop_len
    )


if __name__ == "__main__":
    main()
