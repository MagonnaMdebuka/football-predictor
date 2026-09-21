"""Dixon-Coles match prediction model.

Public API:
    fit_dixon_coles  — fit model to match data, returns FitResult
    build_grid       — build 11x11 score probability grid for a match
    DixonColesParams — fitted parameter dataclass
    FitResult        — fit outcome dataclass
    ScoreGrid        — score probability matrix dataclass
    optimise_xi      — grid-search for optimal time-decay rate
"""
