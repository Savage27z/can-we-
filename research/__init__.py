"""Research platform: run any strategy over any instrument, cost it, and score it.

    strategy   -> signals (where to enter, stop, target)         research/strategy.py
    exits      -> how each trade ends                            research/exits.py
    costs      -> what trading it would have cost                research/costs.py
    engine     -> strategy x instrument x exit x cost -> trades  research/engine.py
    scoring    -> R statistics, drawdown, confidence interval    research/scoring.py

The live bot (live/, tgbot/) is separate and untouched by this package.
"""
