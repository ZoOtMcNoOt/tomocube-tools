"""
Runtime configuration for tomocube-tools.

Provides global settings that can be modified at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TextIO
import sys


@dataclass
class RuntimeConfig:
    """Global runtime configuration."""
    
    verbose: bool = False
    """Enable verbose output for debugging registration and alignment."""
    
    output: TextIO = field(default_factory=lambda: sys.stderr)
    """Output stream for verbose messages."""
    
    def vprint(self, *args, **kwargs) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(*args, file=self.output, **kwargs)


# Global singleton instance
_config = RuntimeConfig()


def get_config() -> RuntimeConfig:
    """Get the global runtime configuration."""
    return _config


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose mode."""
    _config.verbose = enabled


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _config.verbose


def vprint(*args, **kwargs) -> None:
    """Print message only if verbose mode is enabled."""
    _config.vprint(*args, **kwargs)
