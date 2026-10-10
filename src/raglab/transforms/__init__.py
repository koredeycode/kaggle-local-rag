"""Query transforms and the relevance gate (§12, §11).

The four transforms change *what is searched* and are off by default: they
earn their place through ablation deltas or they get removed. The gate is not one
of them — it changes *whether the system speaks at all*, which is a correctness
property, so it defaults on.
"""
