"""Run analytical and dynamics models over a configured parameter range."""

import argparse
import importlib.util
import pickle
import sys
from decimal import Decimal
from pathlib import Path

import config_LearnHaz as config


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "1parameter_sweep"
DATA_DIR = "data/1parameter_sweep/"

SWEEPABLE_PARAMETERS = {
    "PRACTICE_COST",
    "PRACTICE_DECAY",
    "SEVERITY_DECAY",
    "HAZARD_RATE",
    "LEARNING_TIMES"
}


def load_script(filename: str, module_name: str):
    """Load a Python script whose filename may contain hyphens."""
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_sweep_values(start: str, end: str, step: str) -> list[float]:
    """Return start-inclusive values, stopping before passing end."""
    current = Decimal(start)
    stop = Decimal(end)
    increment = Decimal(step)

    if increment == 0:
        raise ValueError("--step must not be zero")
    if (stop > current and increment < 0) or (stop < current and increment > 0):
        raise ValueError("--step must move from --start toward --end")

    values = []
    within_range = (
        (lambda value: value <= stop)
        if increment > 0
        else (lambda value: value >= stop)
    )
    while within_range(current):
        values.append(float(current))
        current += increment

    if not values:
        raise ValueError("The requested range contains no values")
    return values


def resolve_settings(args: argparse.Namespace) -> tuple[str, float, float, float]:
    parameter = args.parameter or getattr(config, "SWEEP_PARAMETER", None)
    start = args.start if args.start is not None else getattr(config, "SWEEP_START", None)
    end = args.end if args.end is not None else getattr(config, "SWEEP_END", None)
    step = args.step if args.step is not None else getattr(config, "SWEEP_STEP", None)

    if parameter is None or start is None or end is None or step is None:
        raise ValueError(
            "Specify --parameter, --start, --end, and --step, or define "
            "SWEEP_PARAMETER, SWEEP_START, SWEEP_END, and SWEEP_STEP "
            "in config_LearnHaz.py."
        )

    parameter = parameter.upper()
    if parameter not in SWEEPABLE_PARAMETERS:
        choices = ", ".join(sorted(SWEEPABLE_PARAMETERS))
        raise ValueError(f"Unsupported parameter {parameter!r}. Choose from: {choices}")

    return parameter, float(start), float(end), float(step)


def parameter_value(parameter: str, value: float, **kwargs) -> str:
    """Format one filename component, replacing the swept value."""
    if parameter in kwargs:
        return f"{parameter}RAN{value:g}-{kwargs[parameter]:g}"
    else:
        return f"{kwargs[parameter]:g}"


def output_filename(model_kind: str, parameter: str, values: list[float]) -> str:
    minimum, maximum = min(values), max(values)
    settings = {
        "PRACTICE_COST": config.PRACTICE_COST,
        "PRACTICE_DECAY": config.PRACTICE_DECAY,
        "SEVERITY_DECAY": config.SEVERITY_DECAY,
        "HAZARD_RATE": config.HAZARD_RATE,
        "LEARNING_TIMES": config.LEARNING_TIMES,
    }

    parts = {}
    for name, default_value in settings.items():
        if name == parameter:
            parts[name] = f"{name}-RAN-{minimum:g}-{maximum:g}"
        else:
            parts[name] = f"{default_value:g}"

    prefix = "HL_analy" if model_kind == "analytical" else "HL_dynam"
    filename = (
        f"{prefix}"
        f"_cost{parts['PRACTICE_COST']}"
        f"_pdecay{parts['PRACTICE_DECAY']}"
        f"_sdecay{parts['SEVERITY_DECAY']}"
        f"_hrate{parts['HAZARD_RATE']}"
        f"_Ltimes{parts['LEARNING_TIMES']}"
        f"_agents{config.N_AGENTS}"
        f"_Nneigh{config.NETWORK_NEIGHBORS}"
        f"_network{config.NETWORK_TYPE}.pkl"
    )

    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", help="Config parameter name to sweep")
    parser.add_argument("--start", help="Range start (inclusive)")
    parser.add_argument("--end", help="Range end")
    parser.add_argument("--step", help="Range increment")
    parser.add_argument("--agents", type=int, default=config.N_AGENTS)
    parser.add_argument("--months", type=int, default=config.DURATION_MONTHS)
    parser.add_argument(
        "--network",
        choices=["small_world", "erdos_renyi", "complete"],
        default=config.NETWORK_TYPE,
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    parameter, start, end, step = resolve_settings(args)
    values = get_sweep_values(str(start), str(end), str(step))
    overrides = {}

    analytical_module = load_script(
        "hazard-learning_analytical.py", "hazard_learning_analytical"
    )
    dynamics_module = load_script(
        "hazard-learning_dynamics.py", "hazard_learning_dynamics_sweep"
    )

    datasets = {"analytical": [], "dynamics": []}
    for value in values:
        overrides = {parameter: value}

        analytical = analytical_module.HazardLearningAnalytical(
            n_agents=args.agents,
            duration_months=args.months,
            network_type=args.network,
            seed=args.seed,
            parameter_overrides=overrides,
        )
        datasets["analytical"].append(
            {"parameter": parameter, "value": value,
             "results": analytical.run_simulation()}
        )

        dynamics = dynamics_module.HazardLearningDynamics(
            n_agents=args.agents,
            duration_months=args.months,
            network_type=args.network,
            seed=args.seed,
            parameter_overrides=overrides,
        )
        datasets["dynamics"].append(
            {"parameter": parameter, "value": value,
             "results": dynamics.run_simulation()}
        )

        print(f"Completed {parameter}={value:g}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_kind, entries in datasets.items():
        path = OUTPUT_DIR / output_filename(model_kind, parameter, values)
        with path.open("wb") as output_file:
            pickle.dump(entries, output_file)
        print(DATA_DIR + output_filename(model_kind, parameter, values) + " \\")


if __name__ == "__main__":
    main()

#python run_parameter_sweep.py --parameter PRACTICE_COST --start 0 --end 2 --step 0.25