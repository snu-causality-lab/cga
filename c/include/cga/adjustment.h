#ifndef CGA_ADJUSTMENT_H
#define CGA_ADJUSTMENT_H

#include "cga/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cga_dag16_adjustment_query cga_dag16_adjustment_query;

/*
 * Complete covariate-adjustment criterion for a known DAG, singleton treatment
 * and singleton outcome, at most 16 vertices. Selectors are vertex BIT MASKS,
 * not indices. No bidirected edges are allowed; acyclicity is checked.
 *
 * Construction caches the forbidden set and proper back-door graph in an owned,
 * immutable snapshot. The source graph may then be mutated or destroyed.
 * Destroy the query once with query_destroy; query_destroy(NULL) is harmless.
 *
 * Empty/overlapping/out-of-range A,Y and NULL arguments: INVALID_ARGUMENT.
 * Non-singleton nonempty disjoint in-range A,Y: UNSUPPORTED_SCOPE.
 * Directed cycle or any bidirected edge: INVALID_GRAPH.
 * Allocation failure: ALLOCATION_FAILED. All errors leave *out_query unchanged.
 * Non-descendant outcomes are supported; no existence or identification outcome
 * is returned by construction. Observability is supplied by the caller's choice
 * of adjustment vertices, not inferred from the graph.
 *
 * Basis: the proper back-door characterization of the complete adjustment
 * criterion (van der Zander, Liskiewicz and Textor, 2014), specialized to DAGs;
 * see docs/reference/DAG_ADJUSTMENT.md for the construction and published sources.
 */
cga_status cga_dag16_adjustment_query_create(
    const cga_directed_bidirected16 *graph,
    uint16_t intervention,
    uint16_t outcome,
    cga_dag16_adjustment_query **out_query);

void cga_dag16_adjustment_query_destroy(cga_dag16_adjustment_query *query);

/*
 * OK true iff adjustment is a valid adjustment set by the complete criterion.
 * OK false is an invalid candidate, not a claim of causal nonidentification.
 * Endpoint/forbidden overlap is an OK false outcome. NULL arguments or bits
 * outside the graph are INVALID_ARGUMENT; errors leave *out_valid unchanged.
 * Performs no allocations. It neither enumerates candidates nor optimizes any
 * variance, and does not require adjustment vertices to precede treatment.
 */
cga_status cga_dag16_adjustment_is_valid(
    const cga_dag16_adjustment_query *query,
    uint16_t adjustment,
    bool *out_valid);

#ifdef __cplusplus
}
#endif

#endif
