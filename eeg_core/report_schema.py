"""Shared EEG screening report constants and JSON-safe validation."""

REPORT_TYPE = "EEG screening report draft"
DIAGNOSTIC_STATUS = "screening_only"
SUPPORTED_REPORT_LANGUAGES = {"zh-CN", "en"}


def validate_language(language: str) -> None:
    """校验 validate language 参数是否合法。"""
    if language not in SUPPORTED_REPORT_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_REPORT_LANGUAGES))
        raise ValueError(f"Unsupported report language '{language}'. Choose from: {supported}.")
