#include "cga/leg.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "LEG contract failure at line %d: %s\n",          \
                    __LINE__, #condition);                                     \
            return 1;                                                          \
        }                                                                      \
    } while (false)

typedef struct {
    size_t count;
    unsigned char marks[16][16];
} snapshot;

static int capture(const cga_pmg16 *graph, snapshot *result)
{
    size_t a;
    size_t b;

    memset(result, 0, sizeof(*result));
    CHECK(cga_pmg16_vertex_count(graph, &result->count) == CGA_STATUS_OK);
    for (a = 0u; a < result->count; ++a) {
        for (b = a + 1u; b < result->count; ++b) {
            cga_endpoint_mark am;
            cga_endpoint_mark bm;

            CHECK(cga_pmg16_get_edge(graph, a, b, &am, &bm) == CGA_STATUS_OK);
            result->marks[a][b] = (unsigned char)am;
            result->marks[b][a] = (unsigned char)bm;
        }
    }
    return 0;
}

static int edge(cga_pmg16 *g, size_t a, cga_endpoint_mark am,
                size_t b, cga_endpoint_mark bm)
{
    CHECK(cga_pmg16_set_edge(g, a, am, b, bm) == CGA_STATUS_OK);
    return 0;
}

static int check_stream(cga_pmg16 *pag, size_t expected)
{
    cga_leg16_iterator *first = NULL;
    cga_leg16_iterator *second = NULL;
    cga_pmg16 *seed = NULL;
    cga_pmg16 *retained = NULL;
    snapshot input_before;
    snapshot input_after;
    snapshot seen[8];
    snapshot original;
    size_t count = 0u;

    CHECK(capture(pag, &input_before) == 0);
    CHECK(cga_pmg16_representative_leg(pag, &seed) == CGA_STATUS_OK);
    CHECK(capture(seed, &original) == 0);
    CHECK(cga_leg16_iterator_create(pag, &first) == CGA_STATUS_OK);
    CHECK(cga_leg16_iterator_create(pag, &second) == CGA_STATUS_OK);
    for (;;) {
        cga_pmg16 *left = NULL;
        cga_pmg16 *right = NULL;
        bool has_left = true;
        bool has_right = false;
        bool valid = false;
        bool equivalent = false;
        snapshot other;
        size_t i;

        CHECK(cga_leg16_iterator_next(first, &left, &has_left) == CGA_STATUS_OK);
        CHECK(cga_leg16_iterator_next(second, &right, &has_right) == CGA_STATUS_OK);
        CHECK(has_left == has_right);
        if (!has_left) {
            CHECK(left == NULL && right == NULL);
            break;
        }
        CHECK(count < 8u);
        CHECK(left != NULL && right != NULL && left != right);
        CHECK(capture(left, &seen[count]) == 0);
        CHECK(capture(right, &other) == 0);
        CHECK(memcmp(&seen[count], &other, sizeof(other)) == 0);
        CHECK(cga_pmg16_is_maximal_ancestral(left, &valid) == CGA_STATUS_OK);
        CHECK(valid);
        CHECK(cga_pmg16_are_markov_equivalent(seed, left, &equivalent) == CGA_STATUS_OK);
        CHECK(equivalent);
        for (i = 0u; i < count; ++i) {
            CHECK(memcmp(&seen[count], &seen[i], sizeof(other)) != 0);
        }
        if (count == 0u) {
            CHECK(memcmp(&seen[0], &original, sizeof(original)) == 0);
            retained = left;
        } else {
            cga_pmg16_destroy(left);
        }
        cga_pmg16_destroy(right);
        ++count;
    }
    CHECK(count == expected);
    {
        cga_pmg16 *result = (cga_pmg16 *)(uintptr_t)1u;
        bool has = true;

        CHECK(cga_leg16_iterator_next(first, &result, &has) == CGA_STATUS_OK);
        CHECK(result == NULL && !has);
    }
    cga_leg16_iterator_destroy(first);
    cga_leg16_iterator_destroy(second);
    CHECK(capture(retained, &input_after) == 0);
    CHECK(memcmp(&input_after, &original, sizeof(original)) == 0);
    CHECK(capture(pag, &input_after) == 0);
    CHECK(memcmp(&input_after, &input_before, sizeof(input_before)) == 0);
    cga_pmg16_destroy(retained);
    cga_pmg16_destroy(seed);
    return 0;
}

static int check_examples(void)
{
    cga_pmg16 *graph = NULL;
    size_t sizes[] = {0u, 1u, 16u};
    size_t i;

    for (i = 0u; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        CHECK(cga_pmg16_create(sizes[i], &graph) == CGA_STATUS_OK);
        CHECK(check_stream(graph, 1u) == 0);
        cga_pmg16_destroy(graph);
    }
    /* A circle chain represents the three noncollider DAGs. */
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_CIRCLE, 1u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(edge(graph, 1u, CGA_ENDPOINT_CIRCLE, 2u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(check_stream(graph, 3u) == 0);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_CIRCLE, 2u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(check_stream(graph, 6u) == 0);
    cga_pmg16_destroy(graph);

    /* Arrowheads at the unshielded collider force one directed LEG. */
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_CIRCLE, 1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 2u, CGA_ENDPOINT_CIRCLE, 1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(check_stream(graph, 1u) == 0);
    cga_pmg16_destroy(graph);

    CHECK(cga_pmg16_create(16u, &graph) == CGA_STATUS_OK);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_CIRCLE, 15u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(edge(graph, 15u, CGA_ENDPOINT_CIRCLE, 14u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(check_stream(graph, 3u) == 0);
    cga_pmg16_destroy(graph);
    return 0;
}

static int check_spouses(void)
{
    cga_pmg16 *graph = NULL;

    /* Valid common-spouse PAG, independently adjudicated by the differential.
     * C=0,A=1,B=2,D=3,E=4; only A-B can reverse. */
    CHECK(cga_pmg16_create(5u, &graph) == CGA_STATUS_OK);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_CIRCLE, 1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_CIRCLE, 2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 1u, CGA_ENDPOINT_CIRCLE, 2u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(edge(graph, 1u, CGA_ENDPOINT_ARROW, 3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 2u, CGA_ENDPOINT_ARROW, 3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 4u, CGA_ENDPOINT_CIRCLE, 3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(check_stream(graph, 2u) == 0);
    cga_pmg16_destroy(graph);
    return 0;
}

static int check_dense_early_stop(void)
{
    cga_pmg16 *pag = NULL;
    cga_pmg16 *first = NULL;
    cga_pmg16 *second = NULL;
    cga_leg16_iterator *iterator = NULL;
    snapshot before;
    snapshot after;
    snapshot other;
    bool has = false;
    size_t a;
    size_t b;

    /* The 16! total orders must not be materialized to obtain two results. */
    CHECK(cga_pmg16_create(16u, &pag) == CGA_STATUS_OK);
    for (a = 0u; a < 16u; ++a) {
        for (b = a + 1u; b < 16u; ++b) {
            CHECK(edge(pag, a, CGA_ENDPOINT_CIRCLE, b, CGA_ENDPOINT_CIRCLE) == 0);
        }
    }
    CHECK(cga_leg16_iterator_create(pag, &iterator) == CGA_STATUS_OK);
    cga_pmg16_destroy(pag);
    CHECK(cga_leg16_iterator_next(iterator, &first, &has) == CGA_STATUS_OK);
    CHECK(has);
    CHECK(capture(first, &before) == 0);
    for (a = 0u; a < 16u; ++a) {
        for (b = a + 1u; b < 16u; ++b) {
            CHECK(before.marks[a][b] == CGA_ENDPOINT_TAIL);
            CHECK(before.marks[b][a] == CGA_ENDPOINT_ARROW);
        }
    }
    CHECK(cga_leg16_iterator_next(iterator, &second, &has) == CGA_STATUS_OK);
    CHECK(has);
    CHECK(capture(second, &other) == 0);
    CHECK(memcmp(&before, &other, sizeof(before)) != 0);
    cga_leg16_iterator_destroy(iterator);
    CHECK(capture(first, &after) == 0);
    CHECK(memcmp(&before, &after, sizeof(before)) == 0);
    cga_pmg16_destroy(first);
    cga_pmg16_destroy(second);
    return 0;
}

static int expect_rejection(cga_pmg16 *graph, cga_status expected)
{
    cga_pmg16 *result = (cga_pmg16 *)(uintptr_t)1u;
    cga_leg16_iterator *iterator = (cga_leg16_iterator *)(uintptr_t)1u;

    CHECK(cga_pmg16_representative_leg(graph, &result) == expected);
    CHECK(result == (cga_pmg16 *)(uintptr_t)1u);
    CHECK(cga_leg16_iterator_create(graph, &iterator) == expected);
    CHECK(iterator == (cga_leg16_iterator *)(uintptr_t)1u);
    return 0;
}

static int check_invalid_inputs(void)
{
    cga_pmg16 *graph = NULL;
    cga_pmg16 *result = (cga_pmg16 *)(uintptr_t)1u;
    cga_leg16_iterator *iterator = NULL;
    bool has = true;
    size_t i;

    CHECK(expect_rejection(NULL, CGA_STATUS_INVALID_ARGUMENT) == 0);
    CHECK(cga_pmg16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(cga_pmg16_representative_leg(graph, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_leg16_iterator_create(graph, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_leg16_iterator_next(NULL, &result, &has) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(result == (cga_pmg16 *)(uintptr_t)1u && has);
    CHECK(cga_leg16_iterator_create(graph, &iterator) == CGA_STATUS_OK);
    CHECK(cga_leg16_iterator_next(iterator, NULL, &has) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(has);
    CHECK(cga_leg16_iterator_next(iterator, &result, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(result == (cga_pmg16 *)(uintptr_t)1u);
    /* Iterator owns its source; a changed and destroyed input is harmless. */
    CHECK(edge(graph, 0u, CGA_ENDPOINT_TAIL, 1u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(expect_rejection(graph, CGA_STATUS_UNSUPPORTED_SCOPE) == 0);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_TAIL, 1u, CGA_ENDPOINT_CIRCLE) == 0);
    CHECK(expect_rejection(graph, CGA_STATUS_UNSUPPORTED_SCOPE) == 0);
    cga_pmg16_destroy(graph);
    CHECK(cga_leg16_iterator_next(iterator, &result, &has) == CGA_STATUS_OK);
    CHECK(has && result != NULL);
    cga_pmg16_destroy(result);
    cga_leg16_iterator_destroy(iterator);
    cga_leg16_iterator_destroy(NULL);

    CHECK(cga_pmg16_create(4u, &graph) == CGA_STATUS_OK);
    for (i = 0u; i < 4u; ++i) {
        CHECK(edge(graph, i, CGA_ENDPOINT_CIRCLE, (i + 1u) % 4u,
                   CGA_ENDPOINT_CIRCLE) == 0);
    }
    CHECK(expect_rejection(graph, CGA_STATUS_INVALID_GRAPH) == 0);
    cga_pmg16_destroy(graph);

    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_TAIL, 1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 1u, CGA_ENDPOINT_TAIL, 2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 2u, CGA_ENDPOINT_TAIL, 0u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(expect_rejection(graph, CGA_STATUS_INVALID_GRAPH) == 0);
    CHECK(edge(graph, 2u, CGA_ENDPOINT_ARROW, 0u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(expect_rejection(graph, CGA_STATUS_INVALID_GRAPH) == 0);
    cga_pmg16_destroy(graph);

    /* Richardson--Spirtes (2002), Fig. 9(a), p. 979: ancestral, nonmaximal.
     * The independent source adjudication is in the maximality contract. */
    CHECK(cga_pmg16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_ARROW, 1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 1u, CGA_ENDPOINT_ARROW, 2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_ARROW, 3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 0u, CGA_ENDPOINT_TAIL, 2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(edge(graph, 1u, CGA_ENDPOINT_TAIL, 3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(expect_rejection(graph, CGA_STATUS_INVALID_GRAPH) == 0);
    cga_pmg16_destroy(graph);
    return 0;
}

int main(void)
{
    CHECK(check_examples() == 0);
    CHECK(check_spouses() == 0);
    CHECK(check_dense_early_stop() == 0);
    CHECK(check_invalid_inputs() == 0);
    puts("LEG contract: passed");
    return 0;
}
