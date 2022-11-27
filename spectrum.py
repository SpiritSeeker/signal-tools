"""
Toolkit for spectrum-related operations with the ADC.

Functions
---------
compute_sndr(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float
    Compute the Signal-to-Noise-and-Distortion Ratio

compute_snr(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float
    Compute the Signal-to-Noise Ratio

compute_sfdr(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float
    Compute the Spurious-Free Dynamic Range

compute_enob(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float
    Compute the Effective Number of Bits
"""

import numpy as np


def _compute_spectrum(
        data: np.ndarray, nfft: int,
        return_onesided: bool = True,
        return_magnitude: bool = True
    ) -> np.ndarray:

    # Pad or truncate signal
    input_data = np.zeros(nfft)
    if data.shape[0] < nfft:
        input_data[:data.shape[0]] = data
    else:
        input_data = data[:nfft]

    # Use Blackman window
    windowed_data = input_data * np.blackman(nfft)

    # Compute FFT
    fft = np.fft.fft(windowed_data, nfft)
    if return_onesided:
        fft = fft[:data.size//2 + 1]
    if return_magnitude:
        fft = np.abs(fft)

    return fft

def compute_sndr(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float:
    """
    Compute the Signal-to-Noise-and-Distortion Ratio.
    Assumes signal to be a sine wave.

    Parameters
    ----------
    data : np.ndarray
        Sine wave data

    nfft : int
        Length of FFT to compute

    **kwargs
        dc_points: int
            Number of points near DC to ignore while computing SNDR
            Default value is `7`.

        signal_bins: int
            Number of FFT bins on either side of the peak to consider as part of signal
            Default value is `12`, implying 25 bins are considered signal bins.

        osr: int
            Over-sampling ratio
            Default value is `1`.
            Noise is integrated till FS / (2 * OSR).

    Returns
    -------
    float
        SNDR of the input signal
    """

    # Get one-sided FFT of the data
    data = np.squeeze(np.array(data))
    spectrum = _compute_spectrum(data, nfft, return_onesided=True, return_magnitude=True)

    # Set values for optional parameters
    dc_points = kwargs.get("dc_points", 7)
    signal_bins = kwargs.get("signal_bins", 12)
    osr = kwargs.get("osr", 1)

    # Remove points close to DC
    spectrum[:dc_points] = 0

    # Compute signal power
    signal_index = np.argmax(spectrum)
    signal_power = np.sum(spectrum[signal_index-signal_bins: signal_index+signal_bins+1] ** 2)

    # Set signal bins to 0
    spectrum[signal_index-signal_bins: signal_index+signal_bins+1] = 0

    # Compute noise power
    noise_power = np.sum(spectrum[:spectrum.size//osr] ** 2)

    return 10 * np.log10(signal_power / (noise_power + np.finfo(np.float32).eps))

def compute_snr(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float:
    """
    Compute the Signal-to-Noise Ratio.
    Assumes signal to be a sine wave.

    Parameters
    ----------
    data : np.ndarray
        Sine wave data

    nfft : int
        Length of FFT to compute

    **kwargs
        dc_points: int
            Number of points near DC to ignore while computing SNDR
            Default value is `7`.

        signal_bins: int
            Number of FFT bins on either side of the peak to consider as part of signal
            Default value is `12`, implying 25 bins are considered signal bins.

        osr: int
            Over-sampling ratio
            Default value is `1`.
            Noise is integrated till FS / (2 * OSR).

        harmonic_bins: int
            Number of FFT bins on either side of the harmonic peak to consider as part of harmonics
            Default value is `3`, implying 7 bins are considered harmonic bins for each harmonic.

        end_harmonic: int
            The highest harmonic to consider as part of distortion
            Default value is `5`.

        include_even_harmonics: bool
            Flag to include even harmonics as part of distortion
            Default value is `False`.

    Returns
    -------
    float
        SNR of the input signal
    """

    # Compute one-sided FFT of the data
    data = np.squeeze(np.array(data))
    spectrum = _compute_spectrum(
        data, nfft, return_onesided=True, return_magnitude=True)

    # Set values of optional parameters
    dc_points = kwargs.get("dc_points", 7)
    signal_bins = kwargs.get("signal_bins", 12)
    osr = kwargs.get("osr", 1)
    harmonic_bins = kwargs.get("harmonic_bins", 3)
    end_harmonic = kwargs.get("end_harmonic", 5)
    include_even_harmonics = kwargs.get("include_even_harmonics", False)

    # Remove points close to DC
    spectrum[:dc_points] = 0

    # Compute signal power
    signal_index = np.argmax(spectrum[dc_points:]) + dc_points
    signal_power = np.sum(
        spectrum[signal_index-signal_bins: signal_index+signal_bins+1] ** 2)
    spectrum[signal_index-signal_bins: signal_index+signal_bins+1] = 0

    # Get harmonic frequency bins
    freqs = signal_index * \
        np.arange(2, end_harmonic+1, step=(1 if include_even_harmonics else 2))

    # Bring harmonic frequencies back to [0:nfft/2]
    freqs = freqs % nfft
    freqs = np.minimum(freqs, nfft-freqs)

    # Set harmonics to 0
    for freq in freqs:
        spectrum[freq-harmonic_bins: freq+harmonic_bins+1] = 0

    # Compute harmonic-free noise power
    noise_power = np.sum(spectrum[:spectrum.size//osr] ** 2)

    return 10 * np.log10(signal_power / (noise_power + np.finfo(np.float32).eps))

def compute_sfdr(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float:
    """
    Compute the Spurious-Free Dynamic Range.
    Assumes signal to be a sine wave.

    Parameters
    ----------
    data : np.ndarray
        Sine wave data

    nfft : int
        Length of FFT to compute

    **kwargs
        dc_points: int
            Number of points near DC to ignore while computing SNDR
            Default value is `7`.

        signal_bins: int
            Number of FFT bins on either side of the peak to consider as part of signal
            Default value is `12`, implying 25 bins are considered signal bins.

        harmonic_bins: int
            Number of FFT bins on either side of the harmonic peak to consider as part of harmonics
            Default value is `3`, implying 7 bins are considered harmonic bins for each harmonic.

        end_harmonic: int
            The highest harmonic to consider as part of distortion
            Default value is `5`.

        include_even_harmonics: bool
            Flag to include even harmonics as part of distortion
            Default value is `False`.

    Returns
    -------
    float
        SFDR of the input signal
    """

    # Compute one-sided FFT of the data
    data = np.squeeze(np.array(data))
    spectrum = _compute_spectrum(
        data, nfft, return_onesided=True, return_magnitude=True)

    # Set values of optional parameters
    dc_points = kwargs.get("dc_points", 7)
    signal_bins = kwargs.get("signal_bins", 12)
    harmonic_bins = kwargs.get("harmonic_bins", 3)
    end_harmonic = kwargs.get("end_harmonic", 5)
    include_even_harmonics = kwargs.get("include_even_harmonics", False)

    # Remove points close to DC
    spectrum[:dc_points] = 0

    # Compute signal power
    signal_index = np.argmax(spectrum[dc_points:]) + dc_points
    signal_power = np.sum(
        spectrum[signal_index-signal_bins: signal_index+signal_bins+1] ** 2)
    spectrum[signal_index-signal_bins: signal_index+signal_bins+1] = 0

    # Get harmonic frequency bins
    freqs = signal_index * \
        np.arange(2, end_harmonic+1, step=(1 if include_even_harmonics else 2))

    # Bring harmonic frequencies back to [0:nfft/2]
    freqs = freqs % nfft
    freqs = np.minimum(freqs, nfft-freqs)

    # Compute maximum harmonic power
    harmonic_power = 0
    for freq in freqs:
        power = np.sum(spectrum[freq-harmonic_bins: freq+harmonic_bins+1] ** 2)
        harmonic_power = max(power, harmonic_power)

    return 10 * np.log10(signal_power / (harmonic_power + np.finfo(np.float32).eps))

def compute_enob(
        data: np.ndarray, nfft: int, **kwargs
    ) -> float:
    """
    Compute the Effective Number of Bits.
    Assumes signal to be a sine wave.

    Parameters
    ----------
    data : np.ndarray
        Sine wave data

    nfft : int
        Length of FFT to compute

    **kwargs
        dc_points: int
            Number of points near DC to ignore while computing SNDR
            Default value is `7`.

        signal_bins: int
            Number of FFT bins on either side of the peak to consider as part of signal
            Default value is `12`, implying 25 bins are considered signal bins.

        osr: int
            Over-sampling ratio
            Default value is `1`.
            Noise is integrated till FS / (2 * OSR).

        harmonic_bins: int
            Number of FFT bins on either side of the harmonic peak to consider as part of harmonics
            Default value is `3`, implying 7 bins are considered harmonic bins for each harmonic.

        end_harmonic: int
            The highest harmonic to consider as part of distortion
            Default value is `5`.

        include_even_harmonics: bool
            Flag to include even harmonics as part of distortion
            Default value is `False`.

    Returns
    -------
    float
        ENOB of the input signal
    """

    snr = compute_snr(data, nfft, **kwargs)

    return (snr - 1.76) / 6.02
