"""Research package for the staged GPU DCOPF sGS-HPR reproduction."""

from .canonical_lp import CanonicalLP
from .dcopf_model import DCOPFConfig, DCOPFModel, build_dcopf_model, load_dcopf_config
from .hpr_generic import HPRResult, HPRState, solve_hpr
from .network_data import NetworkCase, load_matpower_case
from .ptdf import PTDF, build_ptdf
from .residuals import ResidualEvaluation, evaluate_residuals
from .sgs_hpr import (
    SGSHPRResult,
    SGSHPRWorkspace,
    prepare_sgs_hpr,
    sgs_hpr_step,
    solve_sgs_hpr,
)
from .validation import validate_dcopf_candidate

__version__ = "0.0.0"

__all__ = [
    "CanonicalLP",
    "DCOPFConfig",
    "DCOPFModel",
    "HPRResult",
    "HPRState",
    "NetworkCase",
    "PTDF",
    "ResidualEvaluation",
    "SGSHPRResult",
    "SGSHPRWorkspace",
    "build_dcopf_model",
    "build_ptdf",
    "evaluate_residuals",
    "load_dcopf_config",
    "load_matpower_case",
    "prepare_sgs_hpr",
    "sgs_hpr_step",
    "solve_hpr",
    "solve_sgs_hpr",
    "validate_dcopf_candidate",
]
