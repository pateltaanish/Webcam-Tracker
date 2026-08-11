"""Person re-identification (body/clothing appearance).

Used to maintain identity confidence when the target's face isn't visible
(see docs/02_architecture.md sec 3.4 and docs/01_risks_and_assumptions.md
R3). Current implementation (`appearance.py`) is a lightweight HSV
color-histogram signature, not the OSNet embedder docs/02_architecture.md
sec 3.4 eventually selects -- good enough to bridge a short reacquisition gap
without a new dependency; swapping in OSNet later replaces `embed()`'s body
only, not its callers.
"""

from webcam_tracker.reid.appearance import embed, similarity

__all__ = ["embed", "similarity"]
