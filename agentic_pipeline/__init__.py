"""Package agentic_pipeline pour la segmentation et le diagnostic d'incidents."""

__all__ = ["run_agentic_segmentation"]


def __getattr__(name: str):
    # Import paresseux : importer un sous-module (config, concentration...) ne charge pas UMAP ni HDBSCAN
    if name == "run_agentic_segmentation":
        from .pipeline import run_agentic_segmentation

        return run_agentic_segmentation
    raise AttributeError(name)
