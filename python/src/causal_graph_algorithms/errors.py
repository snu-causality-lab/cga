class GraphValidationError(ValueError):
    """Raw graph topology violates a constructor or view invariant."""


class GraphTextSyntaxError(ValueError):
    """Human-readable graph text is malformed."""


class SeparationDomainError(ValueError):
    """A separation query is outside the domain of its published criterion."""


class AncestralDomainError(ValueError):
    """An ancestral-graph operation is outside its published domain."""


class MarkovEquivalenceDomainError(ValueError):
    """A Markov-equivalence query is outside its published domain."""


class PAGDomainError(ValueError):
    """A PAG construction request is outside its published domain."""


class MAGListingDomainError(ValueError):
    """A MAG-listing request is outside its published domain."""


class MAGListingResourceLimitError(RuntimeError):
    """MAG class listing or enumeration exceeds its documented output budget."""
