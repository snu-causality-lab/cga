"""Exact finite SCM and formula interpreters; no planner/ID helpers are imported."""
from fractions import Fraction
from itertools import product


def response(variable, interventions, model, noise):
    nodes, equations, _ = model
    values = {}
    for v in nodes:
        values[v] = interventions[v] if v in interventions else equations[v](values, noise)
    return values[variable]


def term_value(term, model, noise):
    interventions = {a.variable: (a.value.state if hasattr(a.value, "state")
                                  else term_value(a.value, model, noise))
                     for a in term.interventions}
    return response(term.variable, interventions, model, noise)


def event_probability(event, model):
    return sum((p for noise, p in model[2]
                if all(term_value(a.term, model, noise) == a.value.state for a in event.atoms)),
               Fraction())


def kernel_value(kernel, bindings, model):
    interventions = {v: bindings[v] for v in kernel.interventions}
    def marginal(variables):
        return sum((p for noise, p in model[2]
                    if all(response(v, interventions, model, noise) == bindings[v]
                           for v in variables)), Fraction())
    return marginal(kernel.variables + kernel.conditioned_on) / marginal(kernel.conditioned_on)


def formula_value(expression, model, *, experiment_interventions):
    # The evaluator reads the declared syntax; it never normalizes a query or
    # follows a factorization chosen by the production planner. Sources come
    # from explicit test inputs, rather than returned result metadata.
    sources = frozenset(frozenset(s) for s in experiment_interventions)

    def evaluate(node, env):
        name = type(node).__name__
        if name == "ValueConstant":
            return Fraction(node.value)
        if name == "ValueProbability":
            if frozenset(node.kernel.interventions) not in sources:
                raise ValueError("formula requests an unsupplied exact source kernel")
            bindings = {v: (value.state if hasattr(value, "state") else env[value])
                        for v, value in node.bindings}
            return kernel_value(node.kernel, bindings, model)
        if name == "ValueProduct":
            value = Fraction(1)
            for factor in node.factors:
                value *= evaluate(factor, env)
            return value
        if name == "ValueRatio":
            return evaluate(node.numerator, env) / evaluate(node.denominator, env)
        if name == "ValueSummation":
            return sum((evaluate(node.body, env | dict(zip(node.indices, values)))
                        for values in product(*(i.states for i in node.indices))), Fraction())
        raise AssertionError(f"unknown formula node: {name}")

    return evaluate(expression, {})


def binary_noise(probabilities):
    rows = []
    for values in product((0, 1), repeat=len(probabilities)):
        p = Fraction(1)
        for bit, success in zip(values, probabilities):
            p *= success if bit else 1 - success
        rows.append((values, p))
    return tuple(rows)
