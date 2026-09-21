"""Cross-sectional currency strategies, as fixed in research/PREREGISTRATION_XS.md.

Unlike the per-instrument strategies in research/strategies, these rank all sixteen currencies
against each other every Monday, so a decision depends on every currency at once:

    universe   the sixteen currencies and how each is traded against the dollar
    legs       for each Monday cohort and currency: its ranking score and the result of a long
               and a short leg held for the strategy's horizon
    select     which legs a rule picks (momentum, reversal, and the two controls)
    inference  the permutation null, the block bootstrap and the summary statistics
    rules      the pre-registered decision rules, applied by code
    run        the command line
"""
