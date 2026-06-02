"""Model pricing table for cost estimation (USD per million tokens)."""

# (input_per_mtok, output_per_mtok)
_ANTHROPIC_PRICES: dict = {
    # Claude 4 family
    "claude-opus-4":                        (15.00,  75.00),
    "claude-opus-4-8":                      (15.00,  75.00),
    "claude-opus-4-5-20251001":             (15.00,  75.00),
    "claude-sonnet-4":                      ( 3.00,  15.00),
    "claude-sonnet-4-6":                    ( 3.00,  15.00),
    "claude-sonnet-4-5-20251001":           ( 3.00,  15.00),
    "claude-haiku-4":                       ( 0.80,   4.00),
    "claude-haiku-4-5":                     ( 0.80,   4.00),
    "claude-haiku-4-5-20251001":            ( 0.80,   4.00),
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022":           ( 3.00,  15.00),
    "claude-3-5-sonnet-20240620":           ( 3.00,  15.00),
    "claude-3-5-haiku-20241022":            ( 0.80,   4.00),
    # Claude 3 family
    "claude-3-opus-20240229":               (15.00,  75.00),
    "claude-3-sonnet-20240229":             ( 3.00,  15.00),
    "claude-3-haiku-20240307":              ( 0.25,   1.25),
}

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


def _lookup_price(model: str, table: dict):
    prices = table.get(model)
    if prices is None:
        for key, val in table.items():
            if model.startswith(key) or key.startswith(model):
                prices = val
                break
    return prices


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for OpenAI models. Returns 0.0 if unknown."""
    prices = _lookup_price(model, _OPENAI_PRICES)
    if prices is None:
        return 0.0
    return round((input_tokens / 1_000_000) * prices[0] + (output_tokens / 1_000_000) * prices[1], 8)


def estimate_cost_anthropic(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for Anthropic models. Returns 0.0 if unknown."""
    prices = _lookup_price(model, _ANTHROPIC_PRICES)
    if prices is None:
        return 0.0
    return round((input_tokens / 1_000_000) * prices[0] + (output_tokens / 1_000_000) * prices[1], 8)
