"""SHELF CLI subcommand modules."""

from shelf.cli_cmds.models import app as models_app
from shelf.cli_cmds.eval import app as eval_app

__all__ = ["models_app", "eval_app"]
