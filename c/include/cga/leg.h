#ifndef CGA_LEG_H
#define CGA_LEG_H

#include "cga/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cga_leg16_iterator cga_leg16_iterator;

/*
 * Class-level loyal equivalent graphs (LEGs), on 0..16 vertices.
 *
 * CALLER ASSUMPTION: pag is a maximally oriented PAG for a Markov-equivalence
 * class of maximal ancestral graphs without selection variables. This raw
 * topology API does not check PAG closure or certify that assumption. Under
 * it, the results are precisely the class members whose bidirected edges
 * are invariant in the class. A result is not necessarily a LEG relative to
 * an arbitrary chosen MAG, whose directed edges a relative LEG must retain.
 *
 * Checked conditions: endpoint storage is well formed; the circle-circle
 * subgraph is chordal; the completed representative is maximal ancestral.
 * Tail-circle and tail-tail edges return CGA_STATUS_UNSUPPORTED_SCOPE.
 * Nonchordality or an invalid completed MAG returns CGA_STATUS_INVALID_GRAPH.
 * These checks do not establish that a structurally accepted input is a PAG.
 *
 * Construction replaces circle-arrow with tail-arrow and orients circle-
 * circle edges by maximum cardinality search, with ascending vertex ties,
 * without a new unshielded collider in the circle subgraph. Existing tails
 * and arrowheads are retained in the representative. Its MAG classification
 * is a separate check beyond the O(n^2 + m) orientation construction.
 *
 * Source basis: Zhang (2008), Artificial Intelligence, Theorem 2, for the
 * representative-MAG construction; Zhang and Spirtes (2005), Lemma 2 and
 * Theorem 2, for legitimate directed reversals (equal spouse sets and
 * Pa(b) = Pa(a) union {a} for a -> b) and connectivity. The repository's
 * enumeration adaptation uses reverse search with the first lexicographic
 * root-disagreeing legitimate reversal as parent. No equivalence-class list
 * is precomputed. See docs/native.md for exact sources and scope.
 *
 * Every graph returned is newly allocated and caller-owned; destroy it with
 * cga_pmg16_destroy. A non-OK status leaves all output arguments unchanged.
 */
cga_status cga_pmg16_representative_leg(
    const cga_pmg16 *pag,
    cga_pmg16 **out_graph);

/*
 * Copies the input into an independent iterator. The caller may then change
 * or destroy pag. Iteration is deterministic for fixed vertex labels and
 * uses O(n^2 + m) working memory regardless of the number of results.
 */
cga_status cga_leg16_iterator_create(
    const cga_pmg16 *pag,
    cga_leg16_iterator **out_iterator);

/*
 * Returns the representative first, then each remaining class LEG once.
 * Exhaustion is CGA_STATUS_OK with *out_graph = NULL and
 * *out_has_graph = false, including repeated calls after exhaustion.
 * Otherwise *out_has_graph = true. A non-OK status also leaves the iterator
 * unchanged, so allocation failure can be retried without losing a result.
 * No work continues between calls; destroy supports stopping at any point.
 */
cga_status cga_leg16_iterator_next(
    cga_leg16_iterator *iterator,
    cga_pmg16 **out_graph,
    bool *out_has_graph);

/* NULL is allowed. Returned graphs remain valid after iterator destruction. */
void cga_leg16_iterator_destroy(cga_leg16_iterator *iterator);

#ifdef __cplusplus
}
#endif

#endif
