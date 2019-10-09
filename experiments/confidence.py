import mirdata
import csv
import os
import numpy as np
import scipy


def ikala(track_id, source_sep=False):

    if source_sep:
        confidence_fpath = os.path.join(
            "confidence/separation",
            "iKala",
            "{}_lead_VUIMM.wav.conf.csv".format(track_id),
        )
    else:
        confidence_fpath = "confidence/stems/iKala/{}.wav.conf.csv".format(track_id)

    with open(confidence_fpath, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        raw_times, raw_conf = [], []

        for line in reader:
            raw_times.append(float(line[0]))
            raw_conf.append(float(line[1]))

    track_data = mirdata.ikala.load_track(track_id)

    conf = scipy.interpolate.interp1d(
        np.array(raw_times),
        np.array(raw_conf),
        "linear",
        fill_value=0.0,
        bounds_error=False,
    )(track_data.f0.times)

    conf = conf * (track_data.f0.frequencies > 0).astype("float")

    return conf


def medleydb_pitch(track_id):

    confidence_fpath = "confidence/stems/MedleyDB-Pitch/{}.wav.conf.csv".format(
        track_id
    )

    with open(confidence_fpath, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        raw_times, raw_conf = [], []

        for line in reader:
            raw_times.append(float(line[0]))
            raw_conf.append(float(line[1]))

    track_data = mirdata.medleydb_pitch.load_track(track_id)

    conf = scipy.interpolate.interp1d(
        np.array(raw_times),
        np.array(raw_conf),
        "linear",
        fill_value=0.0,
        bounds_error=False,
    )(track_data.pitch.times)

    conf = conf * (track_data.pitch.frequencies > 0).astype("float")

    return conf
