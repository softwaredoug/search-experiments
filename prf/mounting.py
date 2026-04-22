from pathlib import Path
from typing import Optional

from cheat_at_search.data_dir import mount


DEFAULT_MOUNT_PATH = Path.home() / ".search-experiments" / "cheat-at-search"


def ensure_data_mounted(manual_path: Optional[Path | str] = None) -> Path:
    """Ensure cheat_at_search data is mounted in a local directory."""
    resolved_path = Path(manual_path) if manual_path is not None else DEFAULT_MOUNT_PATH
    mount(use_gdrive=False, manual_path=str(resolved_path))
    return resolved_path
