#include "cga/backend.h"
#include "cga/adjustment.h"
#include "cga/leg.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static size_t allocations_before_failure = 0u;

void cga_test_allocator_allow(size_t successful_allocations)
{
    allocations_before_failure = successful_allocations;
}

void *cga_test_calloc(size_t count, size_t size)
{
    if (allocations_before_failure == 0u) {
        return NULL;
    }
    --allocations_before_failure;
    return calloc(count, size);
}

static int leg_code(const cga_pmg16 *graph, uint32_t *out_code)
{
    size_t a;
    size_t b;
    uint32_t code = 0u;

    for (a = 0u; a < 3u; ++a) {
        for (b = a + 1u; b < 3u; ++b) {
            cga_endpoint_mark am;
            cga_endpoint_mark bm;

            if (cga_pmg16_get_edge(graph, a, b, &am, &bm) != CGA_STATUS_OK) {
                return 1;
            }
            code = (code << 4u) | ((uint32_t)am << 2u) | (uint32_t)bm;
        }
    }
    *out_code = code;
    return 0;
}

static int check_leg_allocation_failure(bool triangle)
{
    cga_pmg16 *pag = NULL;
    cga_leg16_iterator *reference = NULL;
    cga_leg16_iterator *iterator = NULL;
    uint32_t expected[6];
    const size_t expected_count = triangle ? 6u : 3u;
    size_t i;
    bool succeeded = false;

    cga_test_allocator_allow(SIZE_MAX);
    if (cga_pmg16_create(3u, &pag) != CGA_STATUS_OK ||
        cga_pmg16_set_edge(pag, 0u, CGA_ENDPOINT_CIRCLE,
                          1u, CGA_ENDPOINT_CIRCLE) != CGA_STATUS_OK ||
        cga_pmg16_set_edge(pag, 1u, CGA_ENDPOINT_CIRCLE,
                          2u, CGA_ENDPOINT_CIRCLE) != CGA_STATUS_OK) {
        return 1;
    }
    if (triangle && cga_pmg16_set_edge(pag, 0u, CGA_ENDPOINT_CIRCLE,
                                     2u, CGA_ENDPOINT_CIRCLE) != CGA_STATUS_OK) {
        return 1;
    }
    if (cga_leg16_iterator_create(pag, &reference) != CGA_STATUS_OK) {
        return 1;
    }
    for (i = 0u; i < expected_count; ++i) {
        cga_pmg16 *graph = NULL;
        bool has = false;

        if (cga_leg16_iterator_next(reference, &graph, &has) != CGA_STATUS_OK ||
            !has || leg_code(graph, &expected[i]) != 0) {
            return 1;
        }
        cga_pmg16_destroy(graph);
    }
    cga_leg16_iterator_destroy(reference);

    /* Fail at every allocation position until construction first succeeds. */
    for (i = 0u; i < 8u; ++i) {
        cga_pmg16 *graph = (cga_pmg16 *)(uintptr_t)1u;
        cga_status status;

        cga_test_allocator_allow(i);
        status = cga_pmg16_representative_leg(pag, &graph);
        if (status == CGA_STATUS_OK) {
            cga_pmg16_destroy(graph);
            succeeded = true;
            break;
        }
        if (status != CGA_STATUS_ALLOCATION_FAILED ||
            graph != (cga_pmg16 *)(uintptr_t)1u) {
            return 1;
        }
    }
    if (!succeeded) {
        return 1;
    }
    succeeded = false;
    for (i = 0u; i < 8u; ++i) {
        cga_status status;

        iterator = (cga_leg16_iterator *)(uintptr_t)1u;
        cga_test_allocator_allow(i);
        status = cga_leg16_iterator_create(pag, &iterator);
        if (status == CGA_STATUS_OK) {
            succeeded = true;
            break;
        }
        if (status != CGA_STATUS_ALLOCATION_FAILED ||
            iterator != (cga_leg16_iterator *)(uintptr_t)1u) {
            return 1;
        }
    }
    if (!succeeded) {
        return 1;
    }
    cga_pmg16_destroy(pag);
    /* Triangle traversal also fails/retries after DFS backtracking. */
    for (i = 0u; i < expected_count; ++i) {
        cga_pmg16 *graph = (cga_pmg16 *)(uintptr_t)1u;
        bool has = i % 2u == 0u;
        bool before = has;
        uint32_t code = 0u;

        cga_test_allocator_allow(0u);
        if (cga_leg16_iterator_next(iterator, &graph, &has) !=
                CGA_STATUS_ALLOCATION_FAILED ||
            graph != (cga_pmg16 *)(uintptr_t)1u || has != before) {
            return 1;
        }
        cga_test_allocator_allow(1u);
        if (cga_leg16_iterator_next(iterator, &graph, &has) != CGA_STATUS_OK ||
            !has || leg_code(graph, &code) != 0 || code != expected[i]) {
            return 1;
        }
        cga_pmg16_destroy(graph);
    }
    {
        cga_pmg16 *graph = (cga_pmg16 *)(uintptr_t)1u;
        bool has = true;

        cga_test_allocator_allow(0u);
        if (cga_leg16_iterator_next(iterator, &graph, &has) != CGA_STATUS_OK ||
            has || graph != NULL) {
            return 1;
        }
    }
    cga_leg16_iterator_destroy(iterator);
    return 0;
}

static int check_adjustment_allocation_failure(void)
{
    cga_directed_bidirected16 *graph = NULL;
    cga_dag16_adjustment_query *query = NULL;
    size_t count;
    bool succeeded = false;
    bool valid = false;

    cga_test_allocator_allow(SIZE_MAX);
    if (cga_directed_bidirected16_create(3u, &graph) != CGA_STATUS_OK ||
        cga_directed_bidirected16_set_directed(graph, 0u, 1u, true) != CGA_STATUS_OK ||
        cga_directed_bidirected16_set_directed(graph, 0u, 2u, true) != CGA_STATUS_OK) {
        return 1;
    }
    for (count = 0u; count < 8u; ++count) {
        cga_status status;

        query = (cga_dag16_adjustment_query *)(uintptr_t)1u;
        cga_test_allocator_allow(count);
        status = cga_dag16_adjustment_query_create(graph, 1u, 2u, &query);
        if (status == CGA_STATUS_OK) {
            succeeded = true;
            break;
        }
        if (status != CGA_STATUS_ALLOCATION_FAILED ||
            query != (cga_dag16_adjustment_query *)(uintptr_t)1u) {
            return 1;
        }
    }
    cga_directed_bidirected16_destroy(graph);
    if (!succeeded) {
        return 1;
    }
    /* Repeated validity queries must not allocate. */
    cga_test_allocator_allow(0u);
    if (cga_dag16_adjustment_is_valid(query, 0u, &valid) != CGA_STATUS_OK || !valid ||
        cga_dag16_adjustment_is_valid(query, 4u, &valid) != CGA_STATUS_OK || !valid) {
        return 1;
    }
    cga_dag16_adjustment_query_destroy(query);
    return 0;
}

int main(void)
{
    cga_pmg16 *pmg = (cga_pmg16 *)(uintptr_t)1u;
    cga_pmg16 *pmg_result = (cga_pmg16 *)(uintptr_t)1u;
    cga_directed_bidirected16 *directed_bidirected =
        (cga_directed_bidirected16 *)(uintptr_t)1u;
    cga_directed_bidirected16 *directed_bidirected_result =
        (cga_directed_bidirected16 *)(uintptr_t)1u;
    size_t vertex_count = 99u;

    if (cga_pmg16_create(1u, &pmg) != CGA_STATUS_ALLOCATION_FAILED ||
        pmg != (cga_pmg16 *)(uintptr_t)1u) {
        fputs("pmg allocation failure was not transactional\n", stderr);
        return 1;
    }
    if (cga_directed_bidirected16_create(1u, &directed_bidirected) !=
            CGA_STATUS_ALLOCATION_FAILED ||
        directed_bidirected !=
            (cga_directed_bidirected16 *)(uintptr_t)1u) {
        fputs("directed-bidirected allocation failure was not transactional\n",
              stderr);
        return 1;
    }

    cga_test_allocator_allow(1u);
    pmg = NULL;
    if (cga_pmg16_create(1u, &pmg) != CGA_STATUS_OK || pmg == NULL) {
        fputs("could not create PMG allocation-failure fixture\n", stderr);
        return 1;
    }
    if (cga_pmg16_induced_subgraph(
            pmg, UINT16_C(1), &pmg_result) !=
            CGA_STATUS_ALLOCATION_FAILED ||
        pmg_result != (cga_pmg16 *)(uintptr_t)1u ||
        cga_pmg16_vertex_count(pmg, &vertex_count) != CGA_STATUS_OK ||
        vertex_count != 1u) {
        fputs("pmg induced-subgraph allocation failure was not transactional\n",
              stderr);
        cga_pmg16_destroy(pmg);
        return 1;
    }
    cga_pmg16_destroy(pmg);

    cga_test_allocator_allow(1u);
    directed_bidirected = NULL;
    if (cga_directed_bidirected16_create(
            1u, &directed_bidirected) != CGA_STATUS_OK ||
        directed_bidirected == NULL) {
        fputs("could not create DB allocation-failure fixture\n", stderr);
        return 1;
    }
    if (cga_directed_bidirected16_induced_subgraph(
            directed_bidirected,
            UINT16_C(1),
            &directed_bidirected_result) != CGA_STATUS_ALLOCATION_FAILED ||
        directed_bidirected_result !=
            (cga_directed_bidirected16 *)(uintptr_t)1u ||
        cga_directed_bidirected16_vertex_count(
            directed_bidirected, &vertex_count) != CGA_STATUS_OK ||
        vertex_count != 1u) {
        fputs("DB induced-subgraph allocation failure was not transactional\n",
              stderr);
        cga_directed_bidirected16_destroy(directed_bidirected);
        return 1;
    }
    if (cga_directed_bidirected16_latent_projection(
            directed_bidirected,
            UINT16_C(1),
            &directed_bidirected_result) != CGA_STATUS_ALLOCATION_FAILED ||
        directed_bidirected_result !=
            (cga_directed_bidirected16 *)(uintptr_t)1u ||
        cga_directed_bidirected16_vertex_count(
            directed_bidirected, &vertex_count) != CGA_STATUS_OK ||
        vertex_count != 1u) {
        fputs("DB latent-projection allocation failure was not transactional\n",
              stderr);
        cga_directed_bidirected16_destroy(directed_bidirected);
        return 1;
    }
    cga_directed_bidirected16_destroy(directed_bidirected);

    if (check_leg_allocation_failure(false) != 0 ||
        check_leg_allocation_failure(true) != 0) {
        fputs("LEG allocation failure was not transactional\n", stderr);
        return 1;
    }

    if (check_adjustment_allocation_failure() != 0) {
        fputs("DAG adjustment allocation failure was not transactional\n", stderr);
        return 1;
    }

    puts("allocation failure contract: passed");
    return 0;
}
