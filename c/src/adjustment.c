#include "cga/adjustment.h"

#include <stdlib.h>

#define CGA_DAG16_CAPACITY ((size_t)16u)

#ifndef CGA_CALLOC
#define CGA_CALLOC calloc
#endif

struct cga_dag16_adjustment_query {
    cga_directed_bidirected16 *proper_backdoor;
    uint16_t vertex_mask;
    uint16_t intervention;
    uint16_t outcome;
    uint16_t forbidden;
};

static uint16_t cga_adjustment_bit(size_t vertex)
{
    return (uint16_t)(UINT32_C(1) << vertex);
}

static bool cga_adjustment_is_singleton(uint16_t mask)
{
    return mask != UINT16_C(0) &&
        (mask & (uint16_t)(mask - UINT16_C(1))) == UINT16_C(0);
}

cga_status cga_dag16_adjustment_query_create(
    const cga_directed_bidirected16 *graph,
    uint16_t intervention,
    uint16_t outcome,
    cga_dag16_adjustment_query **out_query)
{
    cga_dag16_adjustment_query *query;
    uint16_t children[CGA_DAG16_CAPACITY] = {0};
    uint16_t descendants[CGA_DAG16_CAPACITY] = {0};
    uint16_t vertex_mask;
    uint16_t causal = UINT16_C(0);
    uint16_t forbidden = intervention;
    size_t vertex_count;
    size_t source;
    size_t target;
    size_t treatment_vertex = 0u;
    cga_status status;

    if (graph == NULL || out_query == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    status = cga_directed_bidirected16_vertex_count(graph, &vertex_count);
    if (status != CGA_STATUS_OK) {
        return status;
    }
    if (vertex_count > CGA_DAG16_CAPACITY) {
        return CGA_STATUS_UNSUPPORTED_SCOPE;
    }
    vertex_mask = (uint16_t)((UINT32_C(1) << vertex_count) - UINT32_C(1));
    if (intervention == UINT16_C(0) || outcome == UINT16_C(0) ||
        (intervention & outcome) != UINT16_C(0) ||
        ((intervention | outcome) & (uint16_t)~vertex_mask) != UINT16_C(0)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_adjustment_is_singleton(intervention) ||
        !cga_adjustment_is_singleton(outcome)) {
        return CGA_STATUS_UNSUPPORTED_SCOPE;
    }
    for (source = 0u; source < vertex_count; ++source) {
        if (cga_adjustment_bit(source) == intervention) {
            treatment_vertex = source;
        }
        for (target = 0u; target < vertex_count; ++target) {
            bool present;

            if (source == target) {
                continue;
            }
            status = cga_directed_bidirected16_has_bidirected(graph, source, target, &present);
            if (status != CGA_STATUS_OK) {
                return status;
            }
            if (present) {
                return CGA_STATUS_INVALID_GRAPH;
            }
            status = cga_directed_bidirected16_has_directed(graph, source, target, &present);
            if (status != CGA_STATUS_OK) {
                return status;
            }
            if (present) {
                children[source] |= cga_adjustment_bit(target);
            }
        }
        descendants[source] = children[source];
    }
    /* Strict directed transitive closure; a diagonal bit detects any cycle. */
    for (target = 0u; target < vertex_count; ++target) {
        for (source = 0u; source < vertex_count; ++source) {
            if ((descendants[source] & cga_adjustment_bit(target)) != UINT16_C(0)) {
                descendants[source] |= descendants[target];
            }
        }
    }
    for (source = 0u; source < vertex_count; ++source) {
        const uint16_t source_bit = cga_adjustment_bit(source);

        if ((descendants[source] & source_bit) != UINT16_C(0)) {
            return CGA_STATUS_INVALID_GRAPH;
        }
        /* Singleton A: cn(A,Y) = strict de(A) intersect reflexive an(Y). */
        if ((descendants[treatment_vertex] & source_bit) != UINT16_C(0) &&
            ((descendants[source] | source_bit) & outcome) != UINT16_C(0)) {
            causal |= source_bit;
        }
    }
    for (source = 0u; source < vertex_count; ++source) {
        const uint16_t source_bit = cga_adjustment_bit(source);

        if ((causal & source_bit) != UINT16_C(0)) {
            forbidden |= descendants[source] | source_bit;
        }
    }

    query = CGA_CALLOC((size_t)1u, sizeof(*query));
    if (query == NULL) {
        return CGA_STATUS_ALLOCATION_FAILED;
    }
    status = cga_directed_bidirected16_create(vertex_count, &query->proper_backdoor);
    if (status != CGA_STATUS_OK) {
        free(query);
        return status;
    }
    for (source = 0u; source < vertex_count; ++source) {
        uint16_t retained = children[source];

        /* Remove only the first edge of each proper directed A-Y path. */
        if (source == treatment_vertex) {
            retained &= (uint16_t)~causal;
        }
        for (target = 0u; target < vertex_count; ++target) {
            if ((retained & cga_adjustment_bit(target)) != UINT16_C(0)) {
                status = cga_directed_bidirected16_set_directed(
                    query->proper_backdoor, source, target, true);
                if (status != CGA_STATUS_OK) {
                    cga_dag16_adjustment_query_destroy(query);
                    return status;
                }
            }
        }
    }
    query->vertex_mask = vertex_mask;
    query->intervention = intervention;
    query->outcome = outcome;
    query->forbidden = forbidden;
    *out_query = query;
    return CGA_STATUS_OK;
}

void cga_dag16_adjustment_query_destroy(cga_dag16_adjustment_query *query)
{
    if (query != NULL) {
        cga_directed_bidirected16_destroy(query->proper_backdoor);
        free(query);
    }
}

cga_status cga_dag16_adjustment_is_valid(
    const cga_dag16_adjustment_query *query,
    uint16_t adjustment,
    bool *out_valid)
{
    if (query == NULL || out_valid == NULL ||
        (adjustment & (uint16_t)~query->vertex_mask) != UINT16_C(0)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if ((adjustment & (query->forbidden | query->outcome)) != UINT16_C(0)) {
        *out_valid = false;
        return CGA_STATUS_OK;
    }
    return cga_directed_bidirected16_m_separated(
        query->proper_backdoor, query->intervention, query->outcome, adjustment, out_valid);
}
