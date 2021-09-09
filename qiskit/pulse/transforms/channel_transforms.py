# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from typing import Union

from base_transforms import target_qobj_transform
from qiskit.providers.ibmq import IBMQBackend
from qiskit.pulse import Schedule, Play, MeasureChannel
from qiskit.pulse.channels import PulseChannel
from qiskit.visualization.pulse_v2.events import ChannelEvents

def get_channel_waveform(sched: Schedule, 
                         chan: PulseChannel, 
                         backend: Union[None, IBMQBackend] = None,
                         qubit_index: Union[None, int] = None,
                         chan_freq: Union[None, float] = None,
                         dt: float = 2e-9 / 9, 
                         apply_carrier_wave: bool = False):
    """Returns the flattened waveform for a given :class:`qiskit.pulse.channels.PulseChannel` from an input :class:`qiskit.pulse.Schedule`.

    
    Args:
        sched: A pulse schedule to extract a channel's time-series from

        chan: The PulseChannel on which the waveform is to be returned.
        backend: An IBMQBackend from which the qubit frequency and dt 
            are to be extracted.
        qubit_index: An integer indicating the qubit index.
        chan_freq: A float indicating the channel wave frequency. Not necessary if 
            both backend and qubit_index are specified.
        dt: Qubit drive channel timestep in seconds. Default to the 2/9 ns.
        apply_carrier_wave: Whether the carrier wave is applied to the waveforms.
        
    Returns:
        A complex-valued array of the waveform on the 
            given PulseChannel.

    """
    # Check consistency of arguments
    if not isinstance(chan, PulseChannel):
        raise TypeError("The channel must be a PulseChannel eg., a DriveChannel, " 
                        "ControlChannel or MeasureChannel")

    if apply_carrier_wave:
        if backend is not None and qubit_index is not None:
            if isinstance(chan, MeasureChannel):
                chan_freq = backend.defaults().meas_freq_est[qubit_index]
            else:
                chan_freq = backend.defaults().qubit_freq_est[qubit_index]
        else:
            assert chan_freq is not None

    # Flatten the Schedule and transform it into an iterator of 
    # InstructionTuples
    sched_trans = target_qobj_transform(sched)
    chan_events = ChannelEvents.load_program(sched_trans, chan)
    waveform_inst_tups = chan_events.get_waveforms()
    if backend is not None:
        dt = backend.configuration().dt

    # Bulid the channel waveform
    chan_waveform = np.zeros((sched_trans.duration,), dtype=complex)
    for inst_tup in waveform_inst_tups:
        if isinstance(inst_tup.inst, Play):
            # Unpack the time points and phase and frequency in 
            # the current frame
            t0 = inst_tup.t0
            tf = t0 + inst_tup.inst.duration
            t_array = np.arange(t0, tf) * dt
            phase = inst_tup.frame.phase            
            freq = inst_tup.frame.freq
            
            # Apply phase and frequency shifts and optionally carrier wave
            pulse_waveform = inst_tup.inst.pulse.get_waveform().samples
            pulse_waveform *= np.exp(1j * (2 * np.pi* freq * t_array + phase))

            if apply_carrier_wave:
                pulse_waveform *= np.exp(1j * 2 * np.pi * chan_freq * t_array)

            chan_waveform[t0:tf] += pulse_waveform
    return chan_waveform
