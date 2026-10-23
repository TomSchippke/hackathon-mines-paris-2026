"""Rapport opérationnel HTML statique : actions ciblées, programmes flotte et page méthode."""

__all__ = ["build_report"]


def __getattr__(name: str):
    # Import paresseux : `python -m report` ne charge pandas qu'une fois la CLI analysée
    if name == "build_report":
        from .build import build_report

        return build_report
    raise AttributeError(name)
