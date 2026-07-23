"""Identity fusion (Stage 2.4).

Combines face matching with temporal consistency into a per-track identity:
decides whether a given live track is confidently a specific registered person,
or unknown. Never forces a match to the nearest registered user -- a track is
only confirmed as the target when its face consistently says so, which is what
lets the state machine reacquire the target by identity rather than geometry.
The reid contribution (face-away robustness) is added in a later slice.
"""

from webcam_tracker.identity.factory import create_identity_tracker
from webcam_tracker.identity.tracker import IdentityTracker, TrackIdentity

__all__ = [
    "IdentityTracker",
    "TrackIdentity",
    "create_identity_tracker",
]
