from scipy.signal import savgol_filter


def smooth_signal(signal):

    if len(signal) < 7:
        return signal

    return savgol_filter(signal, 7, 2)