"""Context assembly (§13).

Number the surviving spans `[S1]..[Sn]`, compress each against the query, pack
to the token budget using the generator's own tokenizer, and order so the single
best span lands last — decoder models attend less reliably to what sits in the
middle, and the tail is disproportionately influential.
"""
