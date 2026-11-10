"""Supervised routing of a query: which chamber (class) does it belong to?

    query -> analyze() (same morphology as the documents) -> Naive Bayes -> probability of each chamber

If the winner is sure enough, only that chamber is searched. Otherwise every chamber is searched:
  * no word of the query is known to the model        (nothing to decide on)
  * the best probability is below `min_confidence`    (the query could belong to either chamber)
Searching less is only worth it when the choice is reliable, so a doubtful query keeps full recall.
"""
from dataclasses import dataclass, field

from ..classification.naive_bayes import NaiveBayes


@dataclass
class RouteDecision:
    predicted: str | None
    probabilities: dict = field(default_factory=dict)    # chamber -> probability
    searched: list = field(default_factory=list)         # chambers that will be searched
    reason: str = "confident"                            # confident | low_confidence | no_known_word


class ChamberRouter:
    def __init__(self, model: NaiveBayes, min_confidence: float = 0.8, uniform_priors: bool = False):
        self.model = model
        self.uniform_priors = uniform_priors
        self.min_confidence = min_confidence
        self.chambers = list(model.classes)

    def route(self, terms: list[str]) -> RouteDecision:
        if not self.model.n_known(terms):
            return RouteDecision(None, {}, list(self.chambers), "no_known_word")
        probabilities = self.model.predict_proba(terms, self.uniform_priors)
        best = max(self.chambers, key=lambda c: (probabilities[c], c))
        if probabilities[best] < self.min_confidence:
            return RouteDecision(best, probabilities, list(self.chambers), "low_confidence")
        return RouteDecision(best, probabilities, [best], "confident")
