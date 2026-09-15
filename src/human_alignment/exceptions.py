"""Framework-specific exceptions."""


class AlignmentError(Exception):
    """Base exception for the package."""


class ConfigurationError(AlignmentError):
    """Raised when a component or pipeline configuration is invalid."""


class RegistrationError(AlignmentError):
    """Raised for duplicate or invalid registry entries."""


class UnsupportedSignalError(AlignmentError):
    """Raised when a mechanism cannot consume a supervision representation."""


class AssuranceError(AlignmentError):
    """Raised when an assurance procedure cannot produce a meaningful report."""


class BackendUnavailableError(AlignmentError):
    """Raised when an optional execution backend is not installed or supported."""


class DatasetFormatError(AlignmentError):
    """Raised when training data cannot be normalized to the required representation."""
