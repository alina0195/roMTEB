"""Romanian classification tasks."""

from romteb.tasks.classification.histnero import HistNERoMentionClassification
from romteb.tasks.classification.hatespeech_ro import HateSpeechROClassification
from romteb.tasks.classification.red_v2 import REDv2EmotionClassification
from romteb.tasks.classification.roabsa import RoABSAClassification
from romteb.tasks.classification.ro_offense import RoOffenseClassification
from romteb.tasks.classification.romath import RoMathDomainClassification
from romteb.tasks.classification.saroco import SaRoCoClassification
from romteb.tasks.classification.scitechbanro import SciTechBanROClassification

# Retired (not registered in benchmark.py):
# FakenewsRoClassification, ROFFClassification,
# RoOffenseSequencesClassification, FBRoOffenseClassification.

__all__ = [
    "RoABSAClassification",
    "RoOffenseClassification",
    "HateSpeechROClassification",
    "REDv2EmotionClassification",
    "RoMathDomainClassification",
    "SciTechBanROClassification",
    "SaRoCoClassification",
    "HistNERoMentionClassification",
]
