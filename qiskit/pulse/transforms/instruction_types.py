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

# pylint: disable=invalid-name

"""
Special data types.
"""

from enum import Enum
from typing import Union, List, Optional, NewType, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
from qiskit import pulse
from ..instructions import Instruction


@dataclass
class PhaseFreqTuple:
    """Data to represent a set of frequency and phase values.
    
    Args:
        phase: Phase value in rad.
        freq: Frequency value in Hz.
    """
    phase: float
    freq: float

@dataclass
class BarrierInstruction:
    """Data to represent special pulse instruction of barrier.
    
    Args:
        t0: A time when the instruction is issued.
        dt: System cycle time.
        channels: A list of channels associated with this barrier.
    """
    t0: int
    dt: Optional[float]
    channels: List[pulse.channels.Channel]

@dataclass
class SnapshotInstruction:
    """Data to represent special pulse instruction of snapshot.
    
    Args:
        t0: A time when the instruction is issued.
        dt: System cycle time.
        inst: Snapshot instruction.
    """
    t0: int
    dt: Optional[float]
    inst: pulse.instructions.Snapshot

@dataclass
class ChartAxis:
    """Data to represent an axis information of chart.
    
    Args:
        name: Name of chart.
        channels: Channels associated with chart.
    """
    name: str
    channels: List[pulse.channels.Channel]

@dataclass
class ParsedInstruction:
    """A class to store parsed pulse instructions.
    
    Args:
        t0: A time when the instruction is issued.
        dt: System cycle time.
        frame: A reference frame to run instruction.
        inst: Pulse instruction.
        is_opaque: If there is any unbound parameters.
        xdata: The time point data for plotting.
        ydata: The pulse amplitude data for plotting.
    """
    t0: int
    dt: float
    frame: PhaseFreqTuple 
    inst: Union[Instruction, List[Instruction]]
    is_opaque: bool
    xdata: np.ndarray = None
    ydata: np.ndarray = None

@dataclass
class OpaqueShape:
    """Data to represent a pulse instruction with parameterized shape.
    
    Args:
        duration: Duration of instruction.
        meta: Dictionary containing instruction details.
    """
    duration: np.ndarray
    meta: Dict[str, Any]

@dataclass
class HorizontalAxis:
    """Data to represent configuration of horizontal axis.
    
    Args:
        window: Left and right edge of graph.
        axis_map: Mapping of apparent coordinate system and actual location.
        axis_break_pos: Location of axis break.
        label: Label of horizontal axis.
    """
    window: Tuple[int, int]
    axis_map: Dict[float, Union[float, str]]
    axis_break_pos: List[int]
    label: str


class WaveformType(str, Enum):
    """
    Waveform data type.

    REAL: Assigned to objects that represent real part of waveform.
    IMAG: Assigned to objects that represent imaginary part of waveform.
    OPAQUE: Assigned to objects that represent waveform with unbound parameters.
    """

    REAL = "Waveform.Real"
    IMAG = "Waveform.Imag"
    OPAQUE = "Waveform.Opaque"


class LabelType(str, Enum):
    """
    Label data type.

    PULSE_NAME: Assigned to objects that represent name of waveform.
    PULSE_INFO: Assigned to objects that represent extra info about waveform.
    OPAQUE_BOXTEXT: Assigned to objects that represent box text of opaque shapes.
    CH_NAME: Assigned to objects that represent name of channel.
    CH_SCALE: Assigned to objects that represent scaling factor of channel.
    FRAME: Assigned to objects that represent value of frame.
    SNAPSHOT: Assigned to objects that represent label of snapshot.
    """

    PULSE_NAME = "Label.Pulse.Name"
    PULSE_INFO = "Label.Pulse.Info"
    OPAQUE_BOXTEXT = "Label.Opaque.Boxtext"
    CH_NAME = "Label.Channel.Name"
    CH_INFO = "Label.Channel.Info"
    FRAME = "Label.Frame.Value"
    SNAPSHOT = "Label.Snapshot"


class SymbolType(str, Enum):
    """
    Symbol data type.

    FRAME: Assigned to objects that represent symbol of frame.
    SNAPSHOT: Assigned to objects that represent symbol of snapshot.
    """

    FRAME = "Symbol.Frame"
    SNAPSHOT = "Symbol.Snapshot"


class LineType(str, Enum):
    """
    Line data type.

    BASELINE: Assigned to objects that represent zero line of channel.
    BARRIER: Assigned to objects that represent barrier line.
    """

    BASELINE = "Line.Baseline"
    BARRIER = "Line.Barrier"


class AbstractCoordinate(str, Enum):
    """Abstract coordinate that the exact value depends on the user preference.

    RIGHT: The horizontal coordinate at t0 shifted by the left margin.
    LEFT: The horizontal coordinate at tf shifted by the right margin.
    TOP: The vertical coordinate at the top of chart.
    BOTTOM: The vertical coordinate at the bottom of chart.
    """

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class DynamicString(str, Enum):
    """The string which is dynamically updated at the time of drawing.

    SCALE: A temporal value of chart scaling factor.
    """

    SCALE = "@scale"


class WaveformChannel(pulse.channels.PulseChannel):
    """Dummy channel that doesn't belong to specific pulse channel."""

    prefix = "w"

    def __init__(self):
        """Create new waveform channel."""
        super().__init__(0)


class Plotter(str, Enum):
    """Name of pulse plotter APIs.

    Mpl2D: Matplotlib plotter interface. Show charts in 2D canvas.
    """

    Mpl2D = "mpl2d"


class TimeUnits(str, Enum):
    """Representation of time units.

    SYSTEM_CYCLE_TIME: System time dt.
    NANO_SEC: Nano seconds.
    """

    CYCLES = "dt"
    NS = "ns"


# convenient type to represent union of drawing data
DataTypes = NewType("DataType", Union[WaveformType, LabelType, LineType, SymbolType])

# convenient type to represent union of values to represent a coordinate
Coordinate = NewType("Coordinate", Union[float, AbstractCoordinate])
