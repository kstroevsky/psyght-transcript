"""Backend package for phase1 transcription engines."""

from phase1.backends.base import BackendSpec


def get_backend(backend_id: str):
    from phase1.backends.registry import get_backend as _get_backend

    return _get_backend(backend_id)


def list_backend_ids() -> list[str]:
    from phase1.backends.registry import list_backend_ids as _list_backend_ids

    return _list_backend_ids()

__all__ = ["BackendSpec", "get_backend", "list_backend_ids"]
