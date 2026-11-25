class GraphValidationError(ValueError):
    """Raw graph topology violates a constructor or view invariant."""


class GraphTextSyntaxError(ValueError):
    """Human-readable graph text is malformed."""


class LossyConversionError(ValueError):
    """A requested cross-family conversion cannot preserve the topology."""


class SchemaError(ValueError):
    """Serialized graph input violates the versioned graph schema."""


class SeparationDomainError(ValueError):
    """A separation query is outside the domain of its published criterion."""


class ProjectionDomainError(ValueError):
    """A projection request is outside the domain of its published operation."""


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


class ProbabilityExpressionValidationError(ValueError):
    """A symbolic probability expression violates its structural contract."""


class IdentificationDomainError(ValueError):
    """A causal-effect identification query is outside the algorithm domain."""


class TianAdjustmentDomainError(ValueError):
    """A Tian-to-UCA request is outside the compiler's structural domain."""
