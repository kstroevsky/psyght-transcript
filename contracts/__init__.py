"""Shared contracts used across phases without sharing runtime infrastructure."""

from contracts.transcript import Segment, Transcript, Word, transcript_from_dict, transcript_to_dict

__all__ = [
    "Word",
    "Segment",
    "Transcript",
    "transcript_from_dict",
    "transcript_to_dict",
]
