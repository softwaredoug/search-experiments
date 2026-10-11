from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    yaml_files = sorted(
        path
        for path in configs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )

    errors = []
    for path in yaml_files:
        try:
            list(yaml.compose_all(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            errors.append(f"{path.relative_to(repo_root)}: invalid YAML: {error}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Validated YAML syntax in {len(yaml_files)} config files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
