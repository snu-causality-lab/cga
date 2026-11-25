#include "cga/adjustment.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static uint16_t bit(size_t vertex)
{
    return (uint16_t)(UINT32_C(1) << vertex);
}

static int dump_graph(size_t n, uint64_t code)
{
    cga_directed_bidirected16 *graph = NULL;
    size_t a;
    size_t y;

    if (cga_directed_bidirected16_create(n, &graph) != CGA_STATUS_OK) {
        return 1;
    }
    for (a = 0u; a < n; ++a) {
        for (y = 0u; y < n; ++y) {
            if ((code & (UINT64_C(1) << (a * n + y))) != UINT64_C(0) &&
                cga_directed_bidirected16_set_directed(graph, a, y, true) != CGA_STATUS_OK) {
                cga_directed_bidirected16_destroy(graph);
                return 1;
            }
        }
    }
    printf("%zu\t%" PRIu64 "\t", n, code);
    for (a = 0u; a < n; ++a) {
        for (y = 0u; y < n; ++y) {
            cga_dag16_adjustment_query *query = NULL;
            cga_status status;
            uint32_t z;

            if (a == y) {
                continue;
            }
            status = cga_dag16_adjustment_query_create(graph, bit(a), bit(y), &query);
            if (status == CGA_STATUS_INVALID_GRAPH && a == 0u && y == 1u) {
                puts("I");
                cga_directed_bidirected16_destroy(graph);
                return 0;
            }
            if (status != CGA_STATUS_OK) {
                cga_directed_bidirected16_destroy(graph);
                return 1;
            }
            for (z = UINT32_C(0); z < (UINT32_C(1) << n); ++z) {
                bool valid = false;

                if (((uint16_t)z & (bit(a) | bit(y))) != UINT16_C(0)) {
                    continue;
                }
                if (cga_dag16_adjustment_is_valid(query, (uint16_t)z, &valid) != CGA_STATUS_OK) {
                    cga_dag16_adjustment_query_destroy(query);
                    cga_directed_bidirected16_destroy(graph);
                    return 1;
                }
                putchar(valid ? '1' : '0');
            }
            cga_dag16_adjustment_query_destroy(query);
        }
    }
    putchar('\n');
    cga_directed_bidirected16_destroy(graph);
    return 0;
}

static uint32_t next_random(uint32_t *state)
{
    *state ^= *state << 13u;
    *state ^= *state >> 17u;
    *state ^= *state << 5u;
    return *state;
}

int main(void)
{
    size_t ternary;
    size_t n;
    uint32_t random_state = UINT32_C(20261008);

    /* Every orientation/absence assignment on the six labelled n=4 pairs. */
    for (ternary = 0u; ternary < 729u; ++ternary) {
        size_t remaining = ternary;
        size_t a;
        size_t b;
        uint64_t code = UINT64_C(0);

        for (a = 0u; a < 4u; ++a) {
            for (b = a + 1u; b < 4u; ++b) {
                const size_t choice = remaining % 3u;

                remaining /= 3u;
                if (choice != 0u) {
                    const size_t source = choice == 1u ? a : b;
                    const size_t target = choice == 1u ? b : a;

                    code |= UINT64_C(1) << (source * 4u + target);
                }
            }
        }
        if (dump_graph(4u, code) != 0) {
            return 1;
        }
    }
    /* Seeded random DAGs with shuffled vertex labels and varying density. */
    for (n = 5u; n <= 6u; ++n) {
        size_t sample;

        for (sample = 0u; sample < 32u; ++sample) {
            size_t order[6] = {0u, 1u, 2u, 3u, 4u, 5u};
            size_t a;
            size_t b;
            uint64_t code = UINT64_C(0);

            for (a = n; a > 1u; --a) {
                const size_t selected = (size_t)next_random(&random_state) % a;
                const size_t saved = order[a - 1u];

                order[a - 1u] = order[selected];
                order[selected] = saved;
            }
            for (a = 0u; a < n; ++a) {
                for (b = a + 1u; b < n; ++b) {
                    if ((size_t)next_random(&random_state) % 4u <= sample % 4u) {
                        code |= UINT64_C(1) << (order[a] * n + order[b]);
                    }
                }
            }
            if (dump_graph(n, code) != 0) {
                return 1;
            }
        }
    }
    return 0;
}
