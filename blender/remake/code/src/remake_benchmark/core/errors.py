class BenchmarkError(RuntimeError):
    """Base error for configuration or execution failures."""


class ConfigError(BenchmarkError):
    """Raised when a build or component profile is invalid."""


class JobValidationError(BenchmarkError):
    """Raised when a canonical job violates the schema."""
