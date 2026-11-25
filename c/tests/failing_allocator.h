#ifndef CGA_TEST_FAILING_ALLOCATOR_H
#define CGA_TEST_FAILING_ALLOCATOR_H

#include <stddef.h>

void *cga_test_calloc(size_t count, size_t size);
void cga_test_allocator_allow(size_t successful_allocations);

#endif
