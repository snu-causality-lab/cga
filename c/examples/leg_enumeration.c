#include "cga/leg.h"

#include <stdio.h>

int main(void)
{
    cga_pmg16 *pag = NULL;
    cga_pmg16 *leg = NULL;
    cga_leg16_iterator *iterator = NULL;
    cga_status status;
    size_t count = 0u;
    bool has_graph = false;

    /* The known closed no-selection PAG A o-o B o-o C. */
    status = cga_pmg16_create(3u, &pag);
    if (status != CGA_STATUS_OK) goto done;
    status = cga_pmg16_set_edge(pag, 0u, CGA_ENDPOINT_CIRCLE,
                              1u, CGA_ENDPOINT_CIRCLE);
    if (status != CGA_STATUS_OK) goto done;
    status = cga_pmg16_set_edge(pag, 1u, CGA_ENDPOINT_CIRCLE,
                              2u, CGA_ENDPOINT_CIRCLE);
    if (status != CGA_STATUS_OK) goto done;
    status = cga_leg16_iterator_create(pag, &iterator);
    if (status != CGA_STATUS_OK) goto done;
    for (;;) {
        status = cga_leg16_iterator_next(iterator, &leg, &has_graph);
        if (status != CGA_STATUS_OK || !has_graph) break;
        /* Consume this graph here; no previous outputs need to be retained. */
        ++count;
        cga_pmg16_destroy(leg);
        leg = NULL;
    }
done:
    cga_pmg16_destroy(leg);
    cga_leg16_iterator_destroy(iterator);
    cga_pmg16_destroy(pag);
    if (status != CGA_STATUS_OK) {
        fprintf(stderr, "LEG enumeration: %s\n", cga_status_name(status));
        return 1;
    }
    printf("Circle-chain LEGs: %zu\n", count);
    return count == 3u ? 0 : 1;
}
