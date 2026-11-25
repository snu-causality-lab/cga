#ifndef CGA_BACKEND_H
#define CGA_BACKEND_H

#include "cga/status.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Numeric values are part of ABI version 1. */
typedef enum {
    CGA_BACKEND_PMG16 = 1,
    CGA_BACKEND_DIRECTED_BIDIRECTED16 = 2
} cga_backend_family;

/* Numeric values are part of ABI version 1. */
typedef enum {
    CGA_ENDPOINT_NONE = 0,
    CGA_ENDPOINT_TAIL = 1,
    CGA_ENDPOINT_ARROW = 2,
    CGA_ENDPOINT_CIRCLE = 3
} cga_endpoint_mark;

/* Numeric values are part of ABI version 1; scope is queried per operation. */
typedef enum {
    CGA_OPERATION_M_SEPARATION = 1,
    CGA_OPERATION_ANCESTRAL_CLASSIFICATION = 2,
    CGA_OPERATION_INDUCED_SUBGRAPH = 3,
    CGA_OPERATION_LATENT_PROJECTION = 4,
    CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION = 5,
    CGA_OPERATION_MAG_MARKOV_EQUIVALENCE = 6,
    CGA_OPERATION_REPRESENTATIVE_LEG = 7,
    CGA_OPERATION_LEG_ENUMERATION = 8,
    CGA_OPERATION_DAG_ADJUSTMENT = 9
} cga_backend_operation;

typedef struct {
    uint32_t abi_version;
    size_t max_topology_vertices;
    bool endpoint_marked;
    bool supports_bows;
} cga_backend_capabilities;

/* Backend storage is private and family-specific. */
typedef struct cga_pmg16 cga_pmg16;
typedef struct cga_directed_bidirected16 cga_directed_bidirected16;

/* A non-OK return leaves caller-owned output arguments unchanged. */
cga_status cga_backend_get_capabilities(
    cga_backend_family family,
    cga_backend_capabilities *out_capabilities);
cga_status cga_backend_query_operation(
    cga_backend_family family,
    cga_backend_operation operation,
    bool *out_supported,
    size_t *out_max_vertices);

cga_status cga_pmg16_create(size_t vertex_count, cga_pmg16 **out_graph);
void cga_pmg16_destroy(cga_pmg16 *graph);
cga_status cga_pmg16_vertex_count(
    const cga_pmg16 *graph,
    size_t *out_vertex_count);
cga_status cga_pmg16_set_edge(
    cga_pmg16 *graph,
    size_t first,
    cga_endpoint_mark first_mark,
    size_t second,
    cga_endpoint_mark second_mark);
cga_status cga_pmg16_remove_edge(
    cga_pmg16 *graph,
    size_t first,
    size_t second);
cga_status cga_pmg16_get_edge(
    const cga_pmg16 *graph,
    size_t first,
    size_t second,
    cga_endpoint_mark *out_first_mark,
    cga_endpoint_mark *out_second_mark);
cga_status cga_pmg16_is_ancestral(
    const cga_pmg16 *graph,
    bool *out_ancestral);
cga_status cga_pmg16_is_maximal_ancestral(
    const cga_pmg16 *graph,
    bool *out_maximal_ancestral);
cga_status cga_pmg16_are_markov_equivalent(
    const cga_pmg16 *left,
    const cga_pmg16 *right,
    bool *out_equivalent);
cga_status cga_pmg16_m_separated(
    const cga_pmg16 *graph,
    uint16_t left,
    uint16_t right,
    uint16_t given,
    bool *out_separated);
/*
 * Graph-producing operations allocate a new handle owned by the caller.
 * Retained source indices are compacted in ascending order; the new index of
 * old index i is the number of retained indices strictly smaller than i.
 */
cga_status cga_pmg16_induced_subgraph(
    const cga_pmg16 *source,
    uint16_t retained,
    cga_pmg16 **out_graph);

cga_status cga_directed_bidirected16_create(
    size_t vertex_count,
    cga_directed_bidirected16 **out_graph);
void cga_directed_bidirected16_destroy(cga_directed_bidirected16 *graph);
cga_status cga_directed_bidirected16_vertex_count(
    const cga_directed_bidirected16 *graph,
    size_t *out_vertex_count);
cga_status cga_directed_bidirected16_set_directed(
    cga_directed_bidirected16 *graph,
    size_t source,
    size_t target,
    bool present);
cga_status cga_directed_bidirected16_set_bidirected(
    cga_directed_bidirected16 *graph,
    size_t first,
    size_t second,
    bool present);
cga_status cga_directed_bidirected16_has_directed(
    const cga_directed_bidirected16 *graph,
    size_t source,
    size_t target,
    bool *out_present);
cga_status cga_directed_bidirected16_has_bidirected(
    const cga_directed_bidirected16 *graph,
    size_t first,
    size_t second,
    bool *out_present);
cga_status cga_directed_bidirected16_m_separated(
    const cga_directed_bidirected16 *graph,
    uint16_t left,
    uint16_t right,
    uint16_t given,
    bool *out_separated);
/* These graph outputs use the allocation and compaction contract above. */
cga_status cga_directed_bidirected16_induced_subgraph(
    const cga_directed_bidirected16 *source,
    uint16_t retained,
    cga_directed_bidirected16 **out_graph);
cga_status cga_directed_bidirected16_latent_projection(
    const cga_directed_bidirected16 *source,
    uint16_t retained,
    cga_directed_bidirected16 **out_graph);

#ifdef __cplusplus
}
#endif

#endif
