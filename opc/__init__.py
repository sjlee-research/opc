"""OPC toolkit.

    config.py       settings (dataclasses <- config.yaml)
    layout.py       geometry, rasterization, GDS/PNG I/O, standard test patterns
    litho.py        optics (Hopkins/SOCS), torch simulator, resist, EPE/CD measurements
    correction.py   fragmentation, evaluation (EPE/PV band/MRC), rule-based / model-based OPC, ILT
    pipeline.py     run one method on any layout (clips, Runner, correct_file)
    experiments.py  report stages 1-5 -> outputs/
    viz.py          figures
"""
from .config import Config
from .layout import Grid
from .litho import LithoSimulator, SimResult

__all__ = ["Config", "Grid", "LithoSimulator", "SimResult"]
