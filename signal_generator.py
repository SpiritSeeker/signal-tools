"""
Generation of various common signals.
Each function generates and returns signals as a numpy array

...

Functions
---------
sine_wave(
        n_points: int, sample_rate: float,
        fin: float, vpp: float = 2,
        offset: float = 0, phase: float = 0
    ) -> np.ndarray
    Generate a sine wave
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
            Deafult value is `2`

        offset: float
            Voltage offset of the signal (in V)
            Default value is `0`.

        phase: float
            Phase of the signal (in degrees)
            Default value is `0`.
    """

    vpp = kwargs.get("vpp", 2)
    offset = kwargs.get("offset", 0)
    phase = kwargs.get("phase", 0)

    return offset + (vpp / 2) * np.sin(2 * np.pi *
        ((fin/sample_rate * np.arange(n_points)) + phase/360))
