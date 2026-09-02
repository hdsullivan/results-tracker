"""Paper-ready exports: booktabs LaTeX tables and IEEE-sized matplotlib figures."""

from .latex import ablation_latex, comparison_latex, latex_escape, provenance_note, sweep_latex

__all__ = ["comparison_latex", "ablation_latex", "sweep_latex", "latex_escape", "provenance_note"]
