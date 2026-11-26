# Missing-data graphs

Create a `MissingnessGraph` from a validated DAG or ADMG and a mapping such
as `{"X": "RX"}`. The mapping pairs each partially observed true variable
with its response indicator. All other nodes are fully observed substantive
variables. Omit the deterministic observation proxies, which are leaves
with the true variable and indicator as parents. In an ADMG, represent
unobserved common causes with bidirected edges.

## Recovering the target joint

`mohan_joint_recovery_factors(model)` implements the graphical criterion
and denominator readout for Theorem 2 and Equation (6). Its domain has
no edges between response indicators and no latent parents of indicators.
`None` means that a true variable is a direct parent of its own indicator.
For other cases in the domain it returns one term for each indicator. Each
term names the substantive parents and the indicators that must put the
partially observed parents in their observed state.

To evaluate the functional, divide the complete-case joint cell by the
product of these response probabilities, using positive complete-case
support and the model assumptions. The paper codes the observed state as
zero. The graph API works with vertex names and does not encode the state.
[Mohan, Pearl and Tian (2013), Theorem 2 and Eq. (6)](https://proceedings.neurips.cc/paper/2013/file/0ff8033cf9437c213ee13937b1c4c455-Paper.pdf)

## Recovering the full law

`find_colluding_path(model, "X")` returns a path from the true variable to
its indicator whose interior vertices are all colliders. A direct edge
counts. `missing_data_full_law_obstructions(model)` returns one such path
for each obstructed variable. The criterion concerns the joint distribution
of substantive variables and response indicators under the paper's
missing-data ADMG and positivity assumptions. Some target distributions
are recoverable even when the full law is obstructed.
[Nabi, Bhattacharya and Shpitser (2020), §6](https://proceedings.mlr.press/v119/nabi20a.html)

## Direct testability of conditional independence

`mohan_direct_test_witnesses(model, X, Y, Z)` returns the members of `Y`
that can be varied in a complete-case comparison under the three
conditions of Theorem 1. Response indicators needed to observe a variable
in any of the three sets remain fixed at their observed value. The sets must
be pairwise disjoint. The function checks the syntactic sufficient
criterion. A nonempty result names comparison variables. An empty result
leaves other ways of testing the model open. Graphical implication and
statistical support are checked separately.
[Mohan and Pearl (2014), §3.1](https://proceedings.mlr.press/v33/mohan14.html)

## Binary independence from one partly observed variable

`mohan_binary_independence_interval(first, second)` checks whether the
observed table can be completed so that fully observed binary `X` is
independent of partly observed binary `Y`. Supply one row per value of
`X`, ordered as observed zero, observed one and missing. Use exact counts
or rational probabilities with positive mass in both rows. The returned
interval gives the possible common values of the probability of `Y = 0`.
An empty intersection, returned as `None`, rules out independence.
[Mohan and Pearl (2014), Theorem 3 and Table 2](https://proceedings.mlr.press/v33/mohan14.pdf)
