def camel_to_snake(s: str) -> str:
    return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


def snake_to_camel(s: str) -> str:
    return "".join(word.capitalize() for word in s.split("_"))
