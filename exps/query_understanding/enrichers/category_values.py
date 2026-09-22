from __future__ import annotations


def schema_values(vocabulary: list[str]) -> tuple[list[str], dict[str, str]]:
    values: list[str] = []
    aliases: dict[str, str] = {}
    used = set(vocabulary) | {"Unknown"}
    for index, value in enumerate(vocabulary):
        schema_value = value
        if '"' in value or "\\" in value:
            schema_value = f"__category_{index}__"
            while schema_value in used:
                schema_value = f"_{schema_value}"
            aliases[schema_value] = value
        values.append(schema_value)
    return values, aliases


def append_aliases(prompt: str, field: str, aliases: dict[str, str]) -> str:
    if not aliases:
        return prompt
    options = "\n".join(f"{key} = {value}" for key, value in aliases.items())
    return (
        f"{prompt}\n\nSome {field} values use safe structured-output tokens. "
        f"Use the token for the corresponding value:\n{options}"
    )
