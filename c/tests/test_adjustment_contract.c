#include "cga/adjustment.h"

#include <stdint.h>
#include <stdio.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "line %d: %s\n", __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static uint16_t bit(size_t vertex)
{
    return (uint16_t)(UINT32_C(1) << vertex);
}

static int check_api_errors(void)
{
    cga_directed_bidirected16 *graph = NULL;
    cga_dag16_adjustment_query *query = NULL;
    cga_dag16_adjustment_query *sentinel =
        (cga_dag16_adjustment_query *)(uintptr_t)1u;
    cga_dag16_adjustment_query *output = sentinel;
    bool valid = true;
    bool supported = false;
    size_t capacity = 0u;

    CHECK(cga_backend_query_operation(CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_DAG_ADJUSTMENT, &supported, &capacity) == CGA_STATUS_OK);
    CHECK(supported && capacity == 16u);
    CHECK(cga_backend_query_operation(CGA_BACKEND_PMG16,
              CGA_OPERATION_DAG_ADJUSTMENT, &supported, &capacity) == CGA_STATUS_OK);
    CHECK(!supported && capacity == 0u);
    CHECK(cga_directed_bidirected16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(NULL, bit(0u), bit(1u), &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_dag16_adjustment_query_create(graph, 0u, bit(1u), &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), 0u, &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(0u), &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(4u), bit(1u), &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(4u), &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, 5u, bit(1u), &output) ==
          CGA_STATUS_UNSUPPORTED_SCOPE);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), 6u, &output) ==
          CGA_STATUS_UNSUPPORTED_SCOPE);
    CHECK(output == sentinel);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &query) ==
          CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_is_valid(NULL, 0u, &valid) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(valid);
    CHECK(cga_dag16_adjustment_is_valid(query, 0u, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(4u), &valid) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(valid);
    valid = false;
    CHECK(cga_dag16_adjustment_is_valid(query, bit(15u), &valid) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(!valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(0u), &valid) == CGA_STATUS_OK);
    CHECK(!valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(1u), &valid) == CGA_STATUS_OK);
    CHECK(!valid);
    CHECK(cga_directed_bidirected16_set_bidirected(graph, 2u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &output) ==
          CGA_STATUS_INVALID_GRAPH);
    CHECK(output == sentinel);
    CHECK(cga_directed_bidirected16_set_bidirected(graph, 2u, 3u, false) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 2u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 3u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &output) ==
          CGA_STATUS_INVALID_GRAPH);
    CHECK(output == sentinel);
    /* Mutation, then destruction, of the source cannot invalidate the snapshot. */
    cga_directed_bidirected16_destroy(graph);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(2u), &valid) == CGA_STATUS_OK);
    CHECK(valid);
    cga_dag16_adjustment_query_destroy(query);
    cga_dag16_adjustment_query_destroy(NULL);
    CHECK(cga_directed_bidirected16_create(0u, &graph) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &output) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(output == sentinel);
    cga_directed_bidirected16_destroy(graph);
    return 0;
}

static int check_complete_criterion(void)
{
    cga_directed_bidirected16 *graph = NULL;
    cga_dag16_adjustment_query *query = NULL;
    bool valid = false;
    size_t given;
    /* A -> M -> Y, A -> D; D is a safe treatment descendant. */
    CHECK(cga_directed_bidirected16_create(5u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 0u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 2u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 0u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 2u, 4u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &query) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_is_valid(query, 0u, &valid) == CGA_STATUS_OK && valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(3u), &valid) == CGA_STATUS_OK && valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(2u), &valid) == CGA_STATUS_OK && !valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(4u), &valid) == CGA_STATUS_OK && !valid);
    cga_dag16_adjustment_query_destroy(query);
    cga_directed_bidirected16_destroy(graph);

    /* A -> D <- U -> Y: retaining A -> D is essential when conditioning D. */
    CHECK(cga_directed_bidirected16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 0u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 3u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 3u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &query) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_is_valid(query, 0u, &valid) == CGA_STATUS_OK && valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(2u), &valid) == CGA_STATUS_OK && !valid);
    CHECK(cga_dag16_adjustment_is_valid(query, 12u, &valid) == CGA_STATUS_OK && valid);
    cga_dag16_adjustment_query_destroy(query);
    cga_directed_bidirected16_destroy(graph);

    /* A <- C -> Y: a non-descendant outcome still requires adjustment. */
    CHECK(cga_directed_bidirected16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 2u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 2u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &query) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_is_valid(query, 0u, &valid) == CGA_STATUS_OK && !valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(2u), &valid) == CGA_STATUS_OK && valid);
    cga_dag16_adjustment_query_destroy(query);
    /* Y -> A makes every allowed Z invalid, which is an OK false outcome. */
    CHECK(cga_directed_bidirected16_set_directed(graph, 1u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(0u), bit(1u), &query) == CGA_STATUS_OK);
    for (given = 0u; given < 8u; ++given) {
        CHECK(cga_dag16_adjustment_is_valid(query, (uint16_t)given, &valid) == CGA_STATUS_OK);
        CHECK(!valid);
    }
    cga_dag16_adjustment_query_destroy(query);
    cga_directed_bidirected16_destroy(graph);
    return 0;
}

static int check_sixteen_vertices(void)
{
    cga_directed_bidirected16 *graph = NULL;
    cga_dag16_adjustment_query *query = NULL;
    bool valid = false;

    CHECK(cga_directed_bidirected16_create(16u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 15u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 0u, 14u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(graph, 15u, 13u, true) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_query_create(graph, bit(15u), bit(14u), &query) == CGA_STATUS_OK);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(13u), &valid) == CGA_STATUS_OK && valid);
    CHECK(cga_dag16_adjustment_is_valid(query, bit(0u), &valid) == CGA_STATUS_OK && !valid);
    cga_dag16_adjustment_query_destroy(query);
    cga_directed_bidirected16_destroy(graph);
    return 0;
}

int main(void)
{
    CHECK(check_api_errors() == 0);
    CHECK(check_complete_criterion() == 0);
    CHECK(check_sixteen_vertices() == 0);
    puts("C DAG adjustment contract: passed");
    return 0;
}
