"""Romanian classification tasks."""

from romteb.tasks.classification.histnero import HistNERoMentionClassification
from romteb.tasks.classification.hatespeech_ro import HateSpeechROClassification
from romteb.tasks.classification.red_v2 import REDv2EmotionClassification
from romteb.tasks.classification.roabsa import RoABSAClassification
from romteb.tasks.classification.ro_offense import RoOffenseClassification
from romteb.tasks.classification.romath import RoMathDomainClassification
from romteb.tasks.classification.saroco import SaRoCoClassification
from romteb.tasks.classification.scitechbanro import SciTechBanROClassification
from romteb.tasks.classification.ronli_cls import RoNLIClassification



__all__ = [
    "RoABSAClassification",
    "RoOffenseClassification",
    "HateSpeechROClassification",
    "REDv2EmotionClassification",
    "RoMathDomainClassification",
    "SciTechBanROClassification",
    "SaRoCoClassification",
    "HistNERoMentionClassification",
    "RoNLIClassification"
]
