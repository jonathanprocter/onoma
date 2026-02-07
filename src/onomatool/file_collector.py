import glob
import os


def collect_files(pattern: str) -> list[str]:
    """
    Collect files matching the given glob pattern.

    Args:
        pattern: Glob pattern to match files

    Returns:
        List of file paths matching the pattern
    """
    paths = glob.glob(pattern, recursive=True)
    files = []
    for path in paths:
        if os.path.islink(path):
            continue
        if not os.path.isfile(path):
            continue
        files.append(path)
    return files
