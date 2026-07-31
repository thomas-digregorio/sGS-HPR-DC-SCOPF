"""Research package for the staged GPU DCOPF sGS-HPR reproduction."""

from .canonical_lp import CanonicalLP
from .dcopf_model import DCOPFConfig, DCOPFModel, build_dcopf_model, load_dcopf_config
from .hpr_generic import HPRResult, HPRState, solve_hpr
from .network_data import NetworkCase, load_matpower_case
from .preconditioning import (
    LPPreconditioner,
    NormSummary,
    PreconditioningDiagnostics,
    ScalingIterationDiagnostics,
    precondition_lp,
)
from .ptdf import PTDF, build_ptdf
from .residuals import ResidualEvaluation, evaluate_residuals
from .sgs_hpr import (
    SGSHPRResult,
    SGSHPRWorkspace,
    prepare_sgs_hpr,
    sgs_hpr_step,
    solve_sgs_hpr,
)
from .stage5_control import (
    ResidualSnapshot,
    SigmaUpdate,
    Stage5Control,
    Stage5HistoryEntry,
    Stage5PolicyEvent,
    Stage5SGSHPRResult,
    choose_restart_reasons,
    hprlp_sigma_update,
    hprlp_sigma_update_from_scalars,
    sgs_metric_y_quadratic,
    sgs_restart_merit,
    solve_stage5_sgs_hpr,
)
from .structural_y1 import (
    DCOPFEqualityStructure,
    StructuralY1Diagnostics,
    StructuralY1Solver,
    prepare_dcopf_structural_y1,
    prepare_structural_y1,
)
from .validation import validate_dcopf_candidate

__version__ = "0.0.0"

__all__ = [
    "CanonicalLP",
    "DCOPFConfig",
    "DCOPFEqualityStructure",
    "DCOPFModel",
    "HPRResult",
    "HPRState",
    "LPPreconditioner",
    "NetworkCase",
    "NormSummary",
    "PTDF",
    "PreconditioningDiagnostics",
    "ResidualEvaluation",
    "ResidualSnapshot",
    "SGSHPRResult",
    "SGSHPRWorkspace",
    "ScalingIterationDiagnostics",
    "SigmaUpdate",
    "Stage5Control",
    "Stage5HistoryEntry",
    "Stage5PolicyEvent",
    "Stage5SGSHPRResult",
    "StructuralY1Diagnostics",
    "StructuralY1Solver",
    "build_dcopf_model",
    "build_ptdf",
    "choose_restart_reasons",
    "evaluate_residuals",
    "hprlp_sigma_update",
    "hprlp_sigma_update_from_scalars",
    "load_dcopf_config",
    "load_matpower_case",
    "prepare_dcopf_structural_y1",
    "precondition_lp",
    "prepare_sgs_hpr",
    "prepare_structural_y1",
    "sgs_metric_y_quadratic",
    "sgs_restart_merit",
    "sgs_hpr_step",
    "solve_hpr",
    "solve_sgs_hpr",
    "solve_stage5_sgs_hpr",
    "validate_dcopf_candidate",
]
