"""Shared test utilities."""


def fixture_lines(path) -> list[str]:
    """Read a fixture file, skipping the _meta header line."""
    return [
        line for line in str(path.read_text() if hasattr(path, "read_text") else open(path).read()).strip().split("\n")
        if line.strip() and '"_meta"' not in line
    ]
