import warnings
import numpy as np
import scipy.interpolate


def evaluate(ref_times, ref_freqs, ref_voicing,
             est_times, est_freqs, est_voicing):
    est_freqs_resampled, est_voicing_resampled = resample_melody_series(
        est_times, est_freqs, est_voicing, ref_times, 'linear')

    metrics = {}
    metrics['Overall Accuracy'] = overall_accuracy(
        ref_freqs, ref_voicing, est_freqs_resampled, est_voicing_resampled)
    metrics['Raw Pitch Accuracy'] = raw_pitch_accuracy(
        ref_freqs, ref_voicing, est_freqs_resampled)
    metrics['Raw Chroma Accuracy'] = raw_chroma_accuracy(
        ref_freqs, ref_voicing, est_freqs_resampled)
    metrics['Voicing Recall'] = voicing_recall(
        ref_voicing, est_voicing_resampled)
    # metrics['Voicing Precision'] = voicing_precision(
    #     ref_voicing, est_voicing_resampled)
    metrics['Voicing False Alarm'] = voicing_false_alarm(
        ref_voicing, est_voicing_resampled)
    # metrics['Voicing F-Score'] = (
    #     2.0 * (metrics['Voicing Recall'] * metrics['Voicing Precision']) /
    #     (metrics['Voicing Recall'] + metrics['Voicing Precision']))

    return metrics


def frequency_comparison(est_freqs, ref_freqs, tolerance):
    divisor = np.ones(est_freqs.shape)
    mask = np.logical_and(est_freqs != 0, ref_freqs != 0)
    divisor[mask] = est_freqs[mask] / ref_freqs[mask]
    freq_diff_semitones = np.abs(12.0 * np.log2(divisor))
    freq_diff_semitones[np.logical_or(est_freqs == 0, ref_freqs == 0)] = np.inf
    return freq_diff_semitones < tolerance


def chroma_comparison(est_freqs, ref_freqs, tolerance):
    divisor = np.ones(est_freqs.shape)
    mask = np.logical_and(est_freqs != 0, ref_freqs != 0)
    divisor[mask] = est_freqs[mask] / ref_freqs[mask]
    freq_diff_semitones = np.abs(12.0 * np.log2(divisor))
    freq_diff_semitones[np.logical_or(est_freqs == 0, ref_freqs == 0)] = np.inf
    octave = 12 * np.floor(freq_diff_semitones / 12 + 0.5)
    return np.abs(freq_diff_semitones - octave) < tolerance


def raw_pitch_accuracy(ref_freqs, ref_voicing, est_freqs):
    correct_frequencies = frequency_comparison(est_freqs, ref_freqs, 0.5)
    return np.sum(ref_voicing * correct_frequencies) / np.sum(ref_voicing)


def raw_chroma_accuracy(ref_freqs, ref_voicing, est_freqs):
    correct_frequencies = chroma_comparison(est_freqs, ref_freqs, 0.5)
    return np.sum(ref_voicing * correct_frequencies) / np.sum(ref_voicing)


def voicing_recall(ref_voicing, est_voicing):
    ref_indicator = (ref_voicing > 0).astype(float)
    return np.sum(est_voicing * ref_indicator) / np.sum(ref_indicator)


def voicing_precision(ref_voicing, est_voicing):
    return np.sum(est_voicing * ref_voicing) / np.sum(est_voicing)


def voicing_false_alarm(ref_voicing, est_voicing):
    ref_indicator = (ref_voicing == 0).astype(float)
    return np.sum(est_voicing * ref_indicator) / np.sum(ref_indicator)


# def overall_accuracy(ref_freqs, ref_voicing, est_freqs, est_voicing):
#     correct_frequencies = frequency_comparison(est_freqs, ref_freqs, 0.5)
#     n_points = float(len(ref_freqs))
#     numerator = (ref_voicing * est_voicing * correct_frequencies) + \
#         ((1 - ref_voicing) * (1 - est_voicing))
#     denominator = np.abs(ref_voicing - 0.5) + 0.5
#     return (1. / n_points) * np.sum(numerator / denominator)


def overall_accuracy(ref_freqs, ref_voicing, est_freqs, est_voicing):
    correct_frequencies = frequency_comparison(est_freqs, ref_freqs, 0.5)

    est_weighting = (est_voicing * correct_frequencies +
                     ((1 - est_voicing) * (1 - correct_frequencies)))
    ref_weighting = (ref_voicing * correct_frequencies) + (1 - ref_voicing)

    return np.mean(est_weighting * ref_weighting)


def interp_with_zeros(x, y, kind, x_new):
    # interpolate smoothly
    smooth_resampled = scipy.interpolate.interp1d(
        x, y, kind, fill_value=0.0, bounds_error=False)(x_new)

    # If there's a zero, resample 0's everywhere until the next nonzero value
    resampler_mask = (
        scipy.interpolate.interp1d(
            x, y, 'previous', fill_value=0.0, bounds_error=False)(x_new) *
        scipy.interpolate.interp1d(
            x, y, 'next', fill_value=0.0, bounds_error=False)(x_new)
    )

    return smooth_resampled * (resampler_mask != 0)


def resample_melody_series(times, frequencies, voicing,
                           times_new, kind='linear'):
    """Resamples frequency and voicing time series to a new timescale. Maintains
    any zero ("unvoiced") values in frequencies.
    If ``times`` and ``times_new`` are equivalent, no resampling will be
    performed.

    Parameters
    ----------
    times : np.ndarray
        Times of each frequency value
    frequencies : np.ndarray
        Array of frequency values, >= 0
    voicing : np.ndarray
        Boolean array which indicates voiced or unvoiced
    times_new : np.ndarray
        Times to resample frequency and voicing sequences to
    kind : str
        kind parameter to pass to scipy.interpolate.interp1d.
        (Default value = 'linear')

    Returns
    -------
    frequencies_resampled : np.ndarray
        Frequency array resampled to new timebase
    voicing_resampled : np.ndarray, dtype=bool
        Voicing array resampled to new timebase
    """
    assert kind in ['linear', 'quadratic', 'cubic']

    # If the timebases are already the same, no need to interpolate
    if times.shape == times_new.shape and np.allclose(times, times_new):
        return frequencies, voicing.astype(np.bool)

    # Warn when the delta between the original times is not constant,
    if not np.allclose(np.diff(times), np.diff(times).mean()):
        warnings.warn(
            "Non-uniform timescale passed to resample_melody_series.  Pitch "
            "will be linearly interpolated, which will result in undesirable "
            "behavior if silences are indicated by missing values.  Silences "
            "should be indicated by nonpositive frequency values.")

    # interpolate frequencies
    frequencies_resampled = interp_with_zeros(
        times, frequencies, kind, times_new)
    voicing_resampled = interp_with_zeros(times, voicing, kind, times_new)

    return frequencies_resampled, voicing_resampled
