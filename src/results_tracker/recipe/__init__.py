"""Recipes: declare a Method and a Problem once, then run comparisons, sweeps and ablations from a spec.

    from results_tracker.recipe import Method, Problem, Knob, Study, Arm, run_study

The core has no array-library dependency; `results_tracker.recipe.toy` (numpy) is the worked example
behind `results-tracker recipe demo`. See docs/RECIPES.md.
"""

from .core import Estimate, Instance, Method, Problem, Registry, import_ref, registry
from .declared import declared_registry, export_declarations, load_declarations, save_declarations
from .knobs import KINDS as KNOB_KINDS, Knob, KnobSpace
from .study import (
    KINDS as STUDY_KINDS,
    Ablation,
    Arm,
    Job,
    Report,
    Study,
    StudyObserver,
    Sweep,
    arm_changes,
    condition_grid,
    default_diagnostics_dir,
    expand,
    load_study_classes,
    pending_subset,
    run_study,
    validate_study,
)

__all__ = [
    "Knob", "KnobSpace", "KNOB_KINDS",
    "Instance", "Estimate", "Method", "Problem", "Registry", "registry", "import_ref",
    "Study", "Arm", "Sweep", "Ablation", "Job", "Report", "StudyObserver", "STUDY_KINDS", "arm_changes", "default_diagnostics_dir",
    "validate_study", "condition_grid", "expand", "load_study_classes", "pending_subset", "run_study",
    "export_declarations", "declared_registry", "save_declarations", "load_declarations",
]
