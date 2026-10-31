"""Tests never download model weights on their own. Run `RETAKE_OFFLINE=0 uv run pytest models/final` to let the scorer fetch the cross-encoder
from the Hugging Face Hub and run the tests that need it."""
import os

os.environ.setdefault("RETAKE_OFFLINE", "1")
