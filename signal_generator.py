"""
Generation of various common signals.
Each function generates and returns signals as a numpy array.

...

Functions
---------
sine_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray
    Generate a sine wave

square_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray
    Generate a sqaure wave

sawtooth_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray
    Generate a sawtooth wave

triangular_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray
    Generate a triangular wave
"""

import numpy as np


def sine_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray:
    """
    Generate a sine wave.

    Parameters
    ----------
    n_points : int
        Number of points in sine wave to generate

    sample_rate : float
        Sample rate of the generated signal (in Hz)

    fin : float
        Input frequency of the generated signal (in Hz)

    **kwargs
        vpp: float
            Peak-to-peak voltage of the signal
            Deafult value is `2`.

        offset: float
            Voltage offset of the signal (in V)
            Default value is `0`.

        phase: float
            Phase of the signal (in degrees)
            Default value is `0`.

    Returns
    -------
    np.ndarray
        Array containing the generated sine wave
    """

    vpp = kwargs.get("vpp", 2)
    offset = kwargs.get("offset", 0)
    phase = kwargs.get("phase", 0)

    return offset + (vpp / 2) * np.sin(2 * np.pi *
        ((fin/sample_rate * np.arange(n_points)) + phase/360))

def square_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray:
    """
    Generate a square wave.

    Parameters
    ----------
    n_points : int
        Number of points in square wave to generate

    sample_rate : float
        Sample rate of the generated signal (in Hz)

    fin : float
        Input frequency of the generated signal (in Hz)

    **kwargs
        vpp: float
            Peak-to-peak voltage of the signal
            Deafult value is `1`.

        offset: float
            Voltage offset of the signal (in V)
            Default value is `0.5`.

        phase: float
            Phase of the signal (in degrees)
            Default value is `0`.

    Returns
    -------
    np.ndarray
        Array containing the generated square wave
    """

    vpp = kwargs.get("vpp", 1)
    offset = kwargs.get("offset", 0.5)
    phase = kwargs.get("phase", 0)

    # Generate time points
    time_points = (phase/360 / fin) + np.arange(n_points)/sample_rate

    return (1 - ((2*fin*time_points).astype(np.int32) & 1)) * vpp + (offset - vpp/2)

def sawtooth_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray:
    """
    Generate a sawtooth wave.

    Parameters
    ----------
    n_points : int
        Number of points in sawtooth wave to generate

    sample_rate : float
        Sample rate of the generated signal (in Hz)

    fin : float
        Input frequency of the generated signal (in Hz)

    **kwargs
        vpp: float
            Peak-to-peak voltage of the signal
            Deafult value is `1`.

        offset: float
            Voltage offset of the signal (in V)
            Default value is `0.5`.

        phase: float
            Phase of the signal (in degrees)
            Default value is `0`.

    Returns
    -------
    np.ndarray
        Array containing the generated sawtooth wave
    """

    vpp = kwargs.get("vpp", 1)
    offset = kwargs.get("offset", 0.5)
    phase = kwargs.get("phase", 0)

    # Generate sawtooth waveform between 0 and 2
    sawtooth = (phase/180 + np.arange(n_points) * fin / sample_rate) % 2

    return (sawtooth - 1) * vpp/2 + offset

def triangular_wave(
        n_points: int, sample_rate: float,
        fin: float, **kwargs
    ) -> np.ndarray:
    """
    Generate a triangular wave.

    Parameters
    ----------
    n_points : int
        Number of points in triangular wave to generate

    sample_rate : float
        Sample rate of the generated signal (in Hz)

    fin : float
        Input frequency of the generated signal (in Hz)

    **kwargs
        vpp: float
            Peak-to-peak voltage of the signal
            Deafult value is `1`.

        offset: float
            Voltage offset of the signal (in V)
            Default value is `0.5`.

        phase: float
            Phase of the signal (in degrees)
            Default value is `0`.

    Returns
    -------
    np.ndarray
        Array containing the generated triangular wave
    """

    vpp = kwargs.get("vpp", 1)
    offset = kwargs.get("offset", 0.5)
    phase = kwargs.get("phase", 0)

    # Generate sawtooth waveform between 0 and 2
    sawtooth = (phase/180 + np.arange(n_points) * fin / sample_rate) % 2

    return np.minimum(sawtooth, 2-sawtooth) * vpp + (offset - vpp/2)
