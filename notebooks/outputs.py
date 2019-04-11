import csv
import numpy as np
import os
import mirdata
import scipy


def load_crepe(fpath):
    with open(fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        times, freqs, conf = [], [], []
        for line in reader:
            if line[0] == 'time':
                continue
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            conf.append(float(line[2]))
    return np.array(times), np.array(freqs), np.array(conf)


def crepe_outputs(outputs_path):
    output_dict = {}
    mdb_pitch_index = mirdata.medleydb_pitch.track_ids()
    for tid in mdb_pitch_index:
        output_dict[tid] = os.path.join(
            outputs_path, "crepe", "medleydb-pitch",
            "{}.f0.csv".format(tid))
    return output_dict


def load_deepsalience(fpath):
    with open(fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        times, freqs, conf = [], [], []
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            conf.append(float(line[2]))
    return np.array(times), np.array(freqs), np.array(conf)


def deepsalience_outputs(outputs_path):
    output_dict = {}
    ikala_index = mirdata.ikala.track_ids()
    for tid in ikala_index:
        output_dict[tid] = os.path.join(
            outputs_path, "deepsalience", "ikala",
            "{}_melody2_singlef0.csv".format(tid))
    mdb_mel_index = mirdata.medleydb_melody.track_ids()
    for tid in mdb_mel_index:
        output_dict[tid] = os.path.join(
            outputs_path, "deepsalience", "medleydb-melody",
            "{}_MIX_melody2_singlef0.csv".format(tid))
    orchset_index = mirdata.orchset.track_ids()
    for tid in orchset_index:
        output_dict[tid] = os.path.join(
            outputs_path, "deepsalience", "orchset",
            "{}_melody2_singlef0.csv".format(tid))
    return output_dict


def load_melodia(fpath):
    with open(fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        times, freqs, conf = [], [], []
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            conf.append(float(line[2]))
    conf = np.array(conf)
    # conf[conf > 0] += 0.15
    # negative_conf = conf < 0
    # if np.sum(negative_conf) > 0:
    #     conf[negative_conf] = np.abs(conf[negative_conf])
    #     conf[negative_conf] = (
    #         conf[negative_conf] - np.min(conf[negative_conf]) + 0.05)
    #     conf[negative_conf] /= np.max(conf[negative_conf])
    #     conf[negative_conf] *= 0.15

    return np.array(times), np.array(freqs), conf / np.max(conf)


def melodia_outputs(outputs_path):
    output_dict = {}
    ikala_index = mirdata.ikala.track_ids()
    for tid in ikala_index:
        output_dict[tid] = os.path.join(
            outputs_path, "melodia", "ikala",
            "{}.wav.f0.csv".format(tid))
    mdb_mel_index = mirdata.medleydb_melody.track_ids()
    for tid in mdb_mel_index:
        output_dict[tid] = os.path.join(
            outputs_path, "melodia", "medleydb-melody",
            "{}_MIX.wav.f0.csv".format(tid))
    orchset_index = mirdata.orchset.track_ids()
    for tid in orchset_index:
        output_dict[tid] = os.path.join(
            outputs_path, "melodia", "orchset",
            "{}.wav.f0.csv".format(tid))
    return output_dict


def load_pyin(fpath):
    with open(fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        raw_times, raw_freqs = [], []

        for line in reader:
            raw_times.append(float(line[0]))
            raw_freqs.append(float(line[1]))

    conf_fname = "{}_voicedprob.csv".format(
        "_".join(os.path.basename(fpath).split('.')[0].split('_')[:-1]))
    conf_fpath = os.path.join(os.path.dirname(fpath), conf_fname)
    with open(conf_fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        conf_times, conf_vals = [], []

        for line in reader:
            conf_times.append(float(line[0]))
            conf_vals.append(np.max([float(l) for l in line[1:]]))

    max_time = np.max([np.max(raw_times), np.max(conf_times)])
    times = np.arange(0, max_time, 256. / 44100.)
    freqs = np.zeros(times.shape)
    for tval, fval in zip(raw_times, raw_freqs):
        best_idx = np.argmin(np.abs(times - tval))
        freqs[best_idx] = fval

    conf = np.zeros(times.shape)
    for tval, cval in zip(conf_times, conf_vals):
        best_idx = np.argmin(np.abs(times - tval))
        conf[best_idx] = cval

    return times, freqs, conf / np.max(conf)


def pyin_outputs(outputs_path):
    output_dict = {}
    mdb_pitch_index = mirdata.medleydb_pitch.track_ids()
    for tid in mdb_pitch_index:
        output_dict[tid] = os.path.join(
            outputs_path, "pyin", "medleydb-pitch",
            "{}_vamp_pyin_pyin_smoothedpitchtrack.csv".format(tid))
    return output_dict


CREPE = crepe_outputs("../algorithm_outputs")
DEEP_SALIENCE = deepsalience_outputs("../algorithm_outputs")
MELODIA = melodia_outputs("../algorithm_outputs")
PYIN = pyin_outputs("../algorithm_outputs")
