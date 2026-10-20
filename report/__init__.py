"""Rapport opérationnel HTML statique : plan d'action par voie de remédiation."""

__all__ = ["build_report"]


def __getattr__(name: str):
    # Import paresseux : `python -m report` ne charge pandas qu'une fois la CLI analysée
    if name == "build_report":
        from .build import build_report

        return build_report
    raise AttributeError(name)
