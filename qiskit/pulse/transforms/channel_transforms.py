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

from collections import defaultdict
from typing import Optional, Union, Tuple, Iterable, Iterator, Dict, List

import numpy as np

from qiskit.pulse import channels
from qiskit.providers.basebackend import BaseBackend
from qiskit import pulse, circuit
from qiskit.pulse.instructions import Instruction

from .base_transforms import target_qobj_transform
from .instruction_types import PhaseFreqTuple, OpaqueShape, ParsedInstruction
from .device_info import DrawerBackendInfo, OpenPulseBackendInfo
from ..schedule import Schedule, ScheduleBlock

InstructionSched = Union[Tuple[int, Instruction], Instruction]
ScheduleLike = Union[Schedule, ScheduleBlock, 
                     InstructionSched, Iterable[InstructionSched]]
WaveformInstruction = Union[pulse.Play, pulse.Delay, pulse.Acquire]
FrameInstruction = Union[pulse.SetFrequency, pulse.ShiftFrequency, 
                         pulse.SetPhase, pulse.ShiftPhase]

class ChannelTransforms:
    """Channel transform manager."""

    def __init__(
        self,
        waveforms: Dict[int, pulse.Instruction],
        frames: Dict[int, List[pulse.Instruction]],
        channel: pulse.channels.Channel,
        dt: Optional[int] = None,
        init_phase: Optional[float] = None,
        init_frequency: Optional[float] = None
    ):
        """Creates a new channel transform manager.

        Args:
            waveforms: List of waveforms shown in this channel.
            frames: List of frame change type instructions shown in this channel.
            channel: Channel object associated with this manager.
        """
        self._waveforms = waveforms
        self._frames = frames
        self.channel = channel

        # initial frame
        self._init_phase = init_phase or 0
        self._init_frequency = init_frequency or 0

        # time resolution
        self._dt = dt or 0

    @classmethod
    def load_program(cls, 
                     program: ScheduleLike,
                     channel: pulse.channels.Channel,
                     device: Optional[Union[BaseBackend, OpenPulseBackendInfo]] = None
                     ) -> 'ChannelTransforms':
        """Loads a pulse program represented by ``Schedule``.

        Args:
            program: Target ``Schedule`` to visualize.
            channel: The channel managed by this instance.
            backend: The device to extract parameters from.

        Returns:
            The channel transform manager for the specified channel.
        """
        # flatten the schedule
        program = target_qobj_transform(program)

        waveforms = dict()
        frames = defaultdict(list)

        # parse instructions
        for t0, inst in program.filter(channels=[channel]).instructions:
            if isinstance(inst, WaveformInstruction.__args__):
                if inst.duration == 0:
                    # special case, duration of delay can be zero
                    continue
                waveforms[t0] = inst
            elif isinstance(inst, FrameInstruction.__args__):
                frames[t0].append(inst)
        
        chan_transforms = cls(waveforms, frames, channels)   
        if device is not None:
            if isinstance(device, BaseBackend):
                device = OpenPulseBackendInfo.create_from_backend(device)
            dt = device.dt
            init_frequency = device.get_channel_frequency(channel)
            chan_transforms.set_config(dt=dt, init_frequency=init_frequency)

        return chan_transforms

    def set_config(self,
                   dt: Optional[float] = None,
                   init_frequency: Optional[float] = None,
                   init_phase: Optional[float] = None
                   ) -> None:
        """Sets up system status.

        Args:
            dt: Time resolution in sec.
            init_frequency: Modulation frequency in Hz.
            init_phase: Initial phase in rad.
        """
        self._dt = dt or 1
        self._init_frequency = init_frequency or 0
        self._init_phase = init_phase or 0

    def get_parsed_instructions(self, apply_frequency: bool = False
                                ) -> Iterator[ParsedInstruction]:
        """Returns waveform type instructions with frame."""
        sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0], reverse=True)
        sorted_waveforms = sorted(self._waveforms.items(), key=lambda x: x[0])

        # Bind phase and frequency with instruction
        phase = self._init_phase
        frequency = self._init_frequency
        for t0, inst in sorted_waveforms:
            is_opaque = False

            while len(sorted_frame_changes) > 0 and sorted_frame_changes[-1][0] <= t0:
                _, frame_changes = sorted_frame_changes.pop()
                phase, frequency = self._calculate_current_frame(
                    frame_changes=frame_changes, phase=phase, frequency=frequency
                )

            # Convert parameter expression into float
            if isinstance(phase, circuit.ParameterExpression):
                phase = float(phase.bind({param: 0 for param in phase.parameters}))
            if isinstance(frequency, circuit.ParameterExpression):
                frequency = float(frequency.bind({param: 0 for param in frequency.parameters}))

            frame = PhaseFreqTuple(phase, frequency)

            # Check if pulse has unbound parameters
            if isinstance(inst, pulse.Play):
                is_opaque = inst.pulse.is_parameterized()

            parsed_inst = ParsedInstruction(t0, self._dt, frame, inst, is_opaque)
            parsed_inst = self._parse_waveform(parsed_inst)
            parsed_inst = self._apply_phase(parsed_inst)
            if apply_frequency:
                parsed_inst = self._apply_frequency(parsed_inst)

            yield parsed_inst

    def get_frame_changes(self) -> Iterator[ParsedInstruction]:
        """Returns frame change type instructions with total frame change amount."""
        # TODO: parse parametrised FCs correctly

        sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0])

        phase = self._init_phase
        frequency = self._init_frequency
        for t0, frame_changes in sorted_frame_changes:
            is_opaque = False

            pre_phase = phase
            pre_frequency = frequency
            phase, frequency = self._calculate_current_frame(
                frame_changes=frame_changes, phase=phase, frequency=frequency
            )

            # keep parameter expression to check either phase or frequency is parameterized
            frame = PhaseFreqTuple(phase - pre_phase, frequency - pre_frequency)

            # remove parameter expressions to find if next frame is parameterized
            if isinstance(phase, circuit.ParameterExpression):
                phase = float(phase.bind({param: 0 for param in phase.parameters}))
                is_opaque = True
            if isinstance(frequency, circuit.ParameterExpression):
                frequency = float(frequency.bind({param: 0 for param in frequency.parameters}))
                is_opaque = True

            yield ParsedInstruction(t0, self._dt, frame, frame_changes, is_opaque)

    @staticmethod
    def _calculate_current_frame(
        frame_changes: List[pulse.instructions.Instruction], phase: float, frequency: float
    ) -> Tuple[float, float]:
        """Calculates the current frame from the previous frame.

        If parameter is unbound phase or frequency accumulation with this instruction is skipped.

        Args:
            frame_changes: List of frame change instructions at a specific time.
            phase: Phase of previous frame.
            frequency: Frequency of previous frame.

        Returns:
            Phase and frequency of new frame.
        """

        for frame_change in frame_changes:
            if isinstance(frame_change, pulse.instructions.SetFrequency):
                frequency = frame_change.frequency
            elif isinstance(frame_change, pulse.instructions.ShiftFrequency):
                frequency += frame_change.frequency
            elif isinstance(frame_change, pulse.instructions.SetPhase):
                phase = frame_change.phase
            elif isinstance(frame_change, pulse.instructions.ShiftPhase):
                phase += frame_change.phase

        return phase, frequency

    @staticmethod
    def _calculate_current_frame(
        frame_changes: List[pulse.instructions.Instruction], phase: float, frequency: float
    ) -> Tuple[float, float]:
        """Calculates the current frame from the previous frame.

        If parameter is unbound phase or frequency accumulation with this instruction is skipped.

        Args:
            frame_changes: List of frame change instructions at a specific time.
            phase: Phase of previous frame.
            frequency: Frequency of previous frame.

        Returns:
            Phase and frequency of new frame.
        """

        for frame_change in frame_changes:
            if isinstance(frame_change, pulse.instructions.SetFrequency):
                frequency = frame_change.frequency
            elif isinstance(frame_change, pulse.instructions.ShiftFrequency):
                frequency += frame_change.frequency
            elif isinstance(frame_change, pulse.instructions.SetPhase):
                phase = frame_change.phase
            elif isinstance(frame_change, pulse.instructions.ShiftPhase):
                phase += frame_change.phase

        return phase, frequency

    @staticmethod
    def _parse_waveform(
        parsed_inst: ParsedInstruction,
    ) -> Union[ParsedInstruction]:
        """A helper function that generates an array for the waveform with
        instruction metadata.

        Args:
            data: Instruction data set

        Returns:
            A data source to generate a drawing.
        """
        inst = parsed_inst.inst

        if isinstance(inst, pulse.Play):
            # pulse
            operand = inst.pulse
            if isinstance(operand, pulse.ParametricPulse):
                # parametric pulse
                params = operand.parameters
                duration = params.pop("duration", None)
                if isinstance(duration, circuit.Parameter):
                    duration = None

                if parsed_inst.is_opaque:
                    # parametric pulse with unbound parameter
                    return OpaqueShape(duration=duration)
                else:
                    # fixed shape parametric pulse
                    waveform = operand.get_waveform()
            else:
                # waveform
                waveform = operand
            xdata = np.arange(waveform.duration) + parsed_inst.t0
            ydata = waveform.samples
        elif isinstance(inst, pulse.Delay):
            xdata = np.arange(inst.duration) + parsed_inst.t0
            ydata = np.zeros(inst.duration)
        elif isinstance(inst, pulse.Acquire):
            xdata = np.arange(inst.duration) + parsed_inst.t0
            ydata = np.ones(inst.duration)
        else:
            raise TypeError(
                "Unsupported instruction {inst} by "
                "filled envelope.".format(inst=inst.__class__.__name__)
            )

        parsed_inst.xdata = xdata
        parsed_inst.ydata = ydata
        return parsed_inst

    @staticmethod
    def _apply_phase(parsed_inst: ParsedInstruction):
        """Applies phase shifts to a ``ParsedInstruction``."""
        phase = parsed_inst.frame.phase
        parsed_inst.ydata *= np.exp(1j * phase)
        return parsed_inst

    @staticmethod
    def _apply_frequency(parsed_inst: ParsedInstruction):
        """Applies frequency shifts to a ``ParsedInstruction``."""
        frequency = parsed_inst.frame.freq
        # print(f'frequency = {frequency}')
        xdata = np.asarray(parsed_inst.xdata, dtype=float) * parsed_inst.dt
        # print(f'xdata = {2 * np.pi * frequency * xdata}')
        parsed_inst.ydata *= np.exp(1j * 2 * np.pi * frequency * xdata)
        return parsed_inst