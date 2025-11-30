

from typing import Optional


EXCEPTION_ERROR_CODES = {
    KeyboardInterrupt: 130,
    SyntaxError: 2,
    Exception: 1,
}

DETAILED_EXCEPTION_HANDLING = {
    RuntimeError
}


def detailed_exception(err: Exception) -> int:
    if isinstance(err, RuntimeError):
        if "CUDA error: out of memory" in str(err):
            return 125


def get_exit_code(err: Optional[Exception]) -> int:
    if err is None:
        return 0
    for exception, code in EXCEPTION_ERROR_CODES.items():
        if isinstance(err, exception):
            if exception in DETAILED_EXCEPTION_HANDLING:
                return detailed_exception(err)
            return code
