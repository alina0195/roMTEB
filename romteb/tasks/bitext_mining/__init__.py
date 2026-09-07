"""Romanian bitext-mining tasks.

Note: RoMTEB does NOT add new bitext-mining datasets. Instead it relies on
existing multilingual MTEB tasks (NTREX, Tatoeba, IWSLT2017) restricted to
Romanian. See `romteb/benchmark.py` for the registrations.

FloresBitextMining was removed: in the installed mteb version it only exposes
a 'devtest' split, so it produced no usable score for any model.

Bitext-mining is reported as a separate cross-lingual section and is
excluded from the Overall RoMTEB score.
"""
