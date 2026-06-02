"""Model pricing table for cost estimation (USD per million tokens)."""

# (input_per_mtok, output_per_mtok)
_OPENAI_PRICES: dict = {
    # GPT-4o family
    "gpt-4o":                   (2.50,   10.00),
    "gpt-4o-2024-11-20":        (2.50,   10.00),
    "gpt-4o-2024-08-06":        (2.50,   10.00),
    "gpt-4o-2024-05-13":        (5.00,   15.00),
    "gpt-4o-mini":              (0.15,    0.60),
    "gpt-4o-mini-2024-07-18":   (0.15,    0.60),
    # GPT-4 family
    "gpt-4-turbo":              (10.00,  30.00),
    "gpt-4-turbo-2024-04-09":   (10.00,  30.00),
    "gpt-4":                    (30.00,  60.00),
    "gpt-4-32k":                (60.00, 120.00),
    # GPT-3.5
    "gpt-3.5-turbo":            (0.50,   1.50),
    "gpt-3.5-turbo-0125":       (0.50,   1.50),
    # o1 family
    "o1":                       (15.00,  60.00),
    "o1-2024-12-17":            (15.00,  60.00),
    "o1-mini":                  (3.00,   12.00),
    "o1-mini-2024-09-12":       (3.00,   12.00),
    "o1-preview":               (15.00,  60.00),
    "o1-preview-2024-09-12":    (15.00,  60.00),
    # o3 family
    "o3":                       (10.00,  40.00),
    "o3-mini":                  (1.10,    4.40),
    # o4 family
    "o4-mini":                  (1.10,    4.40),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost. Returns 0.0 if model is unknown."""
    prices = _OPENAI_PRICES.get(model)
    if prices is None:
        # Prefix-match so versioned suffixes like "-2024-11-20" still resolve
        for key, val in _OPENAI_PRICES.items():
            if model.startswith(key) or key.startswith(model):
                prices = val
                break
    if prices is None:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * prices[0]
    output_cost = (output_tokens / 1_000_000) * prices[1]
    return round(input_cost + output_cost, 8)
