/* Correctness tests for the kernels. Exits non-zero on the first failure. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"

#define TEST_BLOCKS 64
#define RELATIVE_TOLERANCE 1e-4

static int failures = 0;

static void expect_close(const char *what, double actual, double expected) {
    double allowed = RELATIVE_TOLERANCE * fmax(1.0, fabs(expected));
    if (fabs(actual - expected) <= allowed) return;
    printf("FAIL %s: got %.9g, expected %.9g\n", what, actual, expected);
    failures++;
}

static const WeightFormat *format_named(const char *name) {
    for (size_t i = 0; i < WEIGHT_FORMAT_COUNT; i++)
        if (strcmp(WEIGHT_FORMATS[i].name, name) == 0) return &WEIGHT_FORMATS[i];
    printf("FAIL no format named %s\n", name);
    exit(1);
}

static void test_e2m1_grid(void) {
    const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int code = 0; code < 8; code++) {
        expect_close("positive E2M1 code", e2m1_value((uint8_t)code), magnitudes[code]);
        expect_close("negative E2M1 code", e2m1_value((uint8_t)(code | 8)), -magnitudes[code]);
    }
}

static void test_e8m0_scale(void) {
    expect_close("E8M0 exponent 127", e8m0_scale(127), 1.0);
    expect_close("E8M0 exponent 129", e8m0_scale(129), 4.0);
    expect_close("E8M0 exponent 126", e8m0_scale(126), 0.5);
}

/* Homework 3, question 3: A = [24, -12, 6, 2, 0, ...] is MXFP4 with scale 4 and elements
 * [6, -3, 1.5, 0.5]; against B = [0.25, 1, -0.5, 2] the dot product is -5. */
static void test_homework_dot_product(const char *format_name) {
    uint8_t elements[WEIGHTS_PER_BLOCK / 2] = {0};
    uint8_t scale = 129;
    float activations[WEIGHTS_PER_BLOCK] = {0.25f, 1.0f, -0.5f, 2.0f};
    elements[0] = (uint8_t)(0x7 | (0xD << 4)); /* 6, then -3 */
    elements[1] = (uint8_t)(0x3 | (0x1 << 4)); /* 1.5, then 0.5 */
    expect_close(format_name, format_named(format_name)->dot(elements, &scale, activations, 1), -5.0);
}

static double reference_dot(const WeightFormat *format, const uint8_t *elements, const uint8_t *scales,
                            const float *activations) {
    double total = 0.0;
    for (size_t block = 0; block < TEST_BLOCKS; block++)
        for (size_t i = 0; i < WEIGHTS_PER_BLOCK; i++)
            total += (double)decoded_weight(format, elements + block * format->element_bytes_per_block,
                                            scales + block * format->scale_bytes_per_block, i)
                   * activations[block * WEIGHTS_PER_BLOCK + i];
    return total;
}

static void test_matches_decoded_reference(const WeightFormat *format) {
    uint8_t *elements = malloc(TEST_BLOCKS * format->element_bytes_per_block);
    uint8_t *scales = malloc(TEST_BLOCKS * (format->scale_bytes_per_block + 1));
    float activations[TEST_BLOCKS * WEIGHTS_PER_BLOCK];
    uint32_t state = 7;
    for (size_t i = 0; i < TEST_BLOCKS * WEIGHTS_PER_BLOCK; i++)
        activations[i] = (float)(next_random(&state) % 2001) / 1000.0f - 1.0f;
    format->fill(elements, scales, TEST_BLOCKS, 42);
    expect_close(format->name, format->dot(elements, scales, activations, TEST_BLOCKS),
                 reference_dot(format, elements, scales, activations));
    free(elements);
    free(scales);
}

static void test_bytes_per_weight(void) {
    expect_close("fp32 bytes", bytes_per_weight(format_named("fp32")), 4.0);
    expect_close("int8 bytes", bytes_per_weight(format_named("int8_fp32_scale")), 1.125);
    expect_close("mxfp4 bytes", bytes_per_weight(format_named("mxfp4")), 0.53125);
}

int main(void) {
    test_e2m1_grid();
    test_e8m0_scale();
    test_homework_dot_product("mxfp4");
    test_homework_dot_product("mxfp4_pair_table");
    for (size_t i = 0; i < WEIGHT_FORMAT_COUNT; i++) test_matches_decoded_reference(&WEIGHT_FORMATS[i]);
    test_bytes_per_weight();
    printf(failures ? "%d FAILED\n" : "all kernel tests passed\n", failures);
    return failures != 0;
}
