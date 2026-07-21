"""Identity fusion.

Combines face-match, reid-match, and temporal consistency into a single
target-confidence score, and decides whether a given track is confidently
the selected registered target, confidently not, or unknown. Never forces a
match to the nearest registered user. Implemented in Stage 2.
"""
