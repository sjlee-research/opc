"""OPC toolkit — single entry point.

    python run.py correct examples/square.gds            # model-based OPC -> outputs/square_opc.gds + .png
    python run.py correct drawing.png --pixel-nm 5       # hand-drawn PNG, 1 pixel = 5 nm
    python run.py simulate examples/square.gds           # no correction, printed contour only
    python run.py assignment                             # INV_X1 / NAND2_X4 Metal1 study -> outputs/assignment/
    python run.py examples                               # 100 example targets -> outputs/examples_100/
    python run.py stage 1 2 3 4 5                        # earlier ArF report experiments -> outputs/report/

Options: -c CONFIG (default: config.yaml; examples/stage default to config_arf193.yaml), --method, --out DIR.
Methods: mb (standard model-based OPC, default), rule, mb_adaptive, mb_hybrid, mb_adaptive_hybrid, mb_pw, ilt.
"""
import argparse
from pathlib import Path

from opc import Config
from opc.experiments import run_stages
from opc.pipeline import METHODS, correct_examples, correct_file, write_example_library

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("correct", "simulate"):
        p = sub.add_parser(name, help="OPC on a layout" if name == "correct" else "simulate a layout without OPC")
        p.add_argument("layout", help="GDS (target on layer 1) or black-and-white PNG")
        if name == "correct":
            p.add_argument("--method", default="mb", choices=[m for m in METHODS if m != "no_opc"])
        p.add_argument("--layer", type=int, default=1)
        p.add_argument("--cell", default=None)
        p.add_argument("--pixel-nm", type=float, default=None, help="nm per pixel for PNG input")
        p.add_argument("--out", default=HERE / "outputs")
    p = sub.add_parser("assignment", help="INV_X1 / NAND2_X4 Metal1: before/after OPC, analysis, tables")
    p.add_argument("--out", default=HERE / "outputs" / "assignment")
    p = sub.add_parser("examples", help="run OPC on every cell of an example library (generated if missing)")
    p.add_argument("--library", default=HERE / "examples" / "examples_100.gds")
    p.add_argument("--method", default="mb", choices=[m for m in METHODS if m != "no_opc"])
    p.add_argument("--regen", action="store_true", help="rebuild the 100-example library first")
    p.add_argument("--out", default=None, help="default: outputs/<library name>")
    p = sub.add_parser("stage", help="report experiments (stages 1-5)")
    p.add_argument("stages", nargs="+", type=int, choices=range(1, 6))
    p.add_argument("--out", default=HERE / "outputs" / "report")
    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return
    default = HERE / ("config_arf193.yaml" if args.cmd in ("stage", "examples") else "config.yaml")
    cfg = Config.from_yaml(args.config or default)
    if args.cmd == "assignment":
        from opc.assignment import run_assignment
        run_assignment(cfg, args.out)
    elif args.cmd == "stage":
        run_stages(cfg, args.stages, args.out)
    elif args.cmd == "examples":
        lib = Path(args.library)
        if args.regen or not lib.exists():
            write_example_library(lib, cfg.patterns.cd_nm)
            print(f"wrote {lib}")
        correct_examples(cfg, lib, args.method, args.out or HERE / "outputs" / lib.stem)
    else:
        method = args.method if args.cmd == "correct" else "no_opc"
        correct_file(cfg, args.layout, method, args.out, args.layer, args.cell, args.pixel_nm)


if __name__ == "__main__":
    main()
