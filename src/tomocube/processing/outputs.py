"""Complete a file before publishing it; never replace an input acquisition."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_output(path, *, sources=(), overwrite=True):
    """Yield a sibling temporary path and publish only after successful writing.

    With overwrite=False, publication uses an exclusive hard link so an output
    created by another process during the write is also preserved.
    """
    destination = Path(path)
    for source in sources:
        source = Path(source)
        if (destination.resolve() == source.resolve()
                or destination.exists() and source.exists() and destination.samefile(source)):
            raise ValueError(f"Output would overwrite input: {source}")
    if not overwrite and destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix,
                                dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def new_output_directory(path):
    """Publish a complete flat sequence into a new or empty directory.

    Existing sequences are preserved: choosing a fresh directory avoids stale
    frames or mixing multiple acquisitions. Only generated temporary files are
    removed after failure, never any existing output directory contents.
    """
    destination = Path(path)
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()
                                 or any(destination.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            destination.rmdir()  # Only succeeds if still empty.
        temporary.rename(destination)
    finally:
        if temporary.exists():
            for generated_file in temporary.iterdir():
                generated_file.unlink()
            temporary.rmdir()
