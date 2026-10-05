"""Videos the pipeline refuses to analyse, for reasons the user can fix by filming again.

Each message is written for the person who filmed the video, so a caller can show it as it
is. Callers catch AnalysisError, the base class, and turn it into whatever they show: a line
on the command line, an HTTP response in the web app.
"""


class AnalysisError(Exception):
    """The video can't be analysed; the message says what to change."""


class NotAVideoError(AnalysisError):
    """The file can't be read as a video at all."""


class NoPersonError(AnalysisError):
    """No person was detected in most of the video, so there is nothing to measure."""


class SingleRepError(AnalysisError):
    """The clip holds one rep, whose start and end cannot be told apart from getting down
    and getting up. Raised rather than guessed; the caller tells the user to film more."""
