"""
Module for providing few common filtering operations.

...

Functions
---------
rc_lowpass_filter(
        data: np.ndarray,
        sample_rate: float,
        resistors: float | np.ndarray,
        capacitors: float | np.ndarray,
        order: int = 1
    ) -> np.ndarray
    Implementation of an N-th order RC low pass filter
"""

from typing import Union

import numpy as np
import scipy.signal


def rc_lowpass_filter(
        data: np.ndarray,
        sample_rate: float,
        resistors: Union[float, np.ndarray],
        capacitors: Union[float, np.ndarray],
        order: int = 1
    ) -> np.ndarray:
    """
    Implementation of an N-th order RC low pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data to be filtered

    sample_rate : np.ndarray
        Sample rate of the input data (in Hz)
        To be used in getting the digital equivalent of the RC low pass filter

    resistors : float | np.ndarray
        Values of resistances in the RC network
        If `order` is greater than 1 and single value of resistance is given,
        the same value is taken for all resistors.

    capacitors : float | np.ndarray
        Values of capacitances in the RC network
        If `order` is greater than 1 and single value of capacitance is given,
        the same value is taken for all capacitors.

    order : int, optional
        Order of the filter
        Default value is `1`.

    Returns
    -------
    np.ndarray
        Filtered signal
    """

    resistors = np.array([resistors])
    capacitors = np.array([capacitors])
    if len(resistors.shape) > 1:
        resistors = np.squeeze(resistors)
    if len(capacitors.shape) > 1:
        capacitors = np.squeeze(capacitors)
    if resistors.size == 1:
        resistors = np.repeat(resistors, order)
    if capacitors.size == 1:
        capacitors = np.repeat(capacitors, order)

    # Build SOS
    n_sections = int(np.ceil(order / 2))
    sos_coefficients = np.zeros([n_sections, 6])
    for i in range(n_sections):
        precompute_factor = 1 / \
            (2 * sample_rate * resistors[2*i] * capacitors[2*i])

        numerator = precompute_factor / (precompute_factor + 1)
        denominator = (precompute_factor - 1) / (precompute_factor + 1)

        if 2*i + 1 == order:
            sos_coefficients[i] = np.array(
                [numerator, numerator, 0, 1, denominator, 0])
        else:
            precompute_factor_2 = 1 / \
                (2 * sample_rate * resistors[2*i+1] * capacitors[2*i+1])
            numerator *= precompute_factor_2 / (precompute_factor_2 + 1)
            denominator_2 = (precompute_factor_2 - 1) / (precompute_factor_2 + 1)

            sos_coefficients[i] = np.array(
                [numerator, 2*numerator, numerator,
                1, denominator+denominator_2, denominator*denominator_2])

    return scipy.signal.sosfilt(sos_coefficients, data)

def rc_highpass_filter(
    data: np.ndarray,
    sample_rate: float,
    resistors: Union[float, np.ndarray],
    capacitors: Union[float, np.ndarray],
    order: int = 1
) -> np.ndarray:
    """
    Implementation of an N-th order RC high pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data to be filtered

    sample_rate : np.ndarray
        Sample rate of the input data (in Hz)
        To be used in getting the digital equivalent of the RC high pass filter

    resistors : float | np.ndarray
        Values of resistances in the RC network
        If `order` is greater than 1 and single value of resistance is given,
        the same value is taken for all resistors.

    capacitors : float | np.ndarray
        Values of capacitances in the RC network
        If `order` is greater than 1 and single value of capacitance is given,
        the same value is taken for all capacitors.

    order : int, optional
        Order of the filter
        Default value is `1`.

    Returns
    -------
    np.ndarray
        Filtered signal
    """

    resistors = np.array([resistors])
    capacitors = np.array([capacitors])
    if len(resistors.shape) > 1:
        resistors = np.squeeze(resistors)
    if len(capacitors.shape) > 1:
        capacitors = np.squeeze(capacitors)
    if resistors.size == 1:
        resistors = np.repeat(resistors, order)
    if capacitors.size == 1:
        capacitors = np.repeat(capacitors, order)

    # Build SOS
    n_sections = int(np.ceil(order / 2))
    sos_coefficients = np.zeros([n_sections, 6])
    for i in range(n_sections):
        precompute_factor = 1 / \
            (2 * sample_rate * resistors[2*i] * capacitors[2*i])

        numerator = 1 / (precompute_factor + 1)
        denominator = (precompute_factor - 1) / (precompute_factor + 1)

        if 2*i + 1 == order:
            sos_coefficients[i] = np.array(
                [numerator, -numerator, 0, 1, denominator, 0])
        else:
            precompute_factor_2 = 1 / \
                (2 * sample_rate * resistors[2*i+1] * capacitors[2*i+1])
            numerator *= 1 / (precompute_factor_2 + 1)
            denominator_2 = (precompute_factor_2 - 1) / \
                (precompute_factor_2 + 1)

            sos_coefficients[i] = np.array(
                [numerator, -2*numerator, numerator,
                 1, denominator+denominator_2, denominator*denominator_2])

    return scipy.signal.sosfilt(sos_coefficients, data)
