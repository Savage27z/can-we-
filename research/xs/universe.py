"""The sixteen currencies, each traded against the US dollar through its USD pair.

Excluded before any return was looked at (PREREGISTRATION_XS.md): HKD and DKK (hard pegs), CNH
(managed, dense history only from 2014) and TRY (no dense history).
"""
CURRENCIES = ("AUD", "EUR", "GBP", "NZD", "CAD", "CHF", "JPY", "SGD", "SEK", "NOK", "PLN", "CZK",
              "HUF", "MXN", "THB", "ZAR")

# Quoted as X_USD: a long position in the pair is long the currency. Every other currency is
# quoted as USD_X, where a long position in the pair is SHORT the currency.
_BASE_QUOTED = frozenset({"AUD", "EUR", "GBP", "NZD"})


def pair_of(currency: str) -> str:
    if currency not in CURRENCIES:
        raise ValueError(f"{currency!r} is not one of the {len(CURRENCIES)} currencies in the universe")
    return f"{currency}_USD" if currency in _BASE_QUOTED else f"USD_{currency}"


def orientation(currency: str) -> int:
    """+1 if being long the pair is being long the currency, -1 if it is being short it."""
    pair_of(currency)                       # validates the name
    return 1 if currency in _BASE_QUOTED else -1


PAIRS = tuple(pair_of(c) for c in CURRENCIES)
