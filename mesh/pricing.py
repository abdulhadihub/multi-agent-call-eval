import logging

logger = logging.getLogger(__name__)

# List prices in USD per 1M tokens, checked 2026-11-14 on the Groq and
# Google AI pricing pages. Free-tier usage costs $0. These values exist
# only for paid-equivalent estimates in the evaluation report.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gemini-2.5-flash": (0.30, 2.50),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model)
    if prices is None:
        logger.warning("No price entry for model %s; using 0.0", model)
        return 0.0
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
