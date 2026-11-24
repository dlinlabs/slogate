import argparse
import sys
from typing import Never


class UsageParser(argparse.ArgumentParser):
    """Reserve exit 2 for measured failures, not command-line syntax errors."""

    def error(self, message: str) -> Never:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")
