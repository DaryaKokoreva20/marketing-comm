import itertools


def generate_param_combinations(params_grid: dict) -> list[dict]:
    keys = list(params_grid.keys())
    values_product = itertools.product(*(params_grid[key] for key in keys))

    combinations = [
        dict(zip(keys, values))
        for values in values_product
    ]

    if not combinations:
        raise ValueError("Не найдено ни одной конфигурации гиперпараметров.")

    return combinations
