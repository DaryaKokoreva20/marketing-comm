def serialize_class_weight(class_weight) -> str:
    if class_weight is None:
        return "—"

    if isinstance(class_weight, str):
        return class_weight

    if isinstance(class_weight, dict):
        return ", ".join(f"{key}:{value}" for key, value in sorted(class_weight.items()))

    return str(class_weight)


def serialize_optional_value(value) -> str:
    if value is None:
        return "—"
    return str(value)


def serialize_class_weights(class_weights) -> list[int] | None:
    if class_weights is None:
        return None
    return list(class_weights)
