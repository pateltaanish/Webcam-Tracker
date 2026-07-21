"""Face recognition.

Face detection (SCRFD) + embedding extraction (ArcFace/MobileFaceNet) +
comparison against registered users' stored embeddings, with an explicit
"unknown" outcome when confidence is insufficient (see
docs/02_architecture.md sec 3.3). Implemented in Stage 2.
"""
