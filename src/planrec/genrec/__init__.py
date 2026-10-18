"""Large generative recommendation mainline.

The legacy ``DirectGenerativeRecommender`` is intentionally not re-exported
here.  This package models an item as a fixed-length semantic-token sequence
and trains a causal language model to generate that sequence.
"""

from planrec.genrec.constraints import TokenSequenceTrie
from planrec.genrec.corpus import GenRecCorpusBuilder
from planrec.genrec.rewards import recommendation_reward
from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer

__all__ = [
    "GenRecCorpusBuilder",
    "ResidualKMeansSemanticTokenizer",
    "TokenSequenceTrie",
    "recommendation_reward",
]
