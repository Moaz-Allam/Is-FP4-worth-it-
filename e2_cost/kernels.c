#include "kernels.h"

#include <math.h>
#include <string.h>

#define CODES_PER_BYTE 2
#define CODE_BITS 4
#define CODE_MASK 0x0F
#define E8M0_BIAS 127
#define SMALLEST_FILLED_EXPONENT 120
#define FILLED_EXPONENT_RANGE 8
#define INT8_FILL_SCALE 0.01f

static const float E2M1_VALUES[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

typedef struct {
    float low;
    float high;
} CodePair;

static CodePair code_pairs[256];
static float e8m0_scales[256];
static int tables_ready = 0;

static void prepare_tables(void) {
    if (tables_ready) return;
    for (int byte = 0; byte < 256; byte++) {
        code_pairs[byte].low = E2M1_VALUES[byte & CODE_MASK];
        code_pairs[byte].high = E2M1_VALUES[byte >> CODE_BITS];
        e8m0_scales[byte] = ldexpf(1.0f, byte - E8M0_BIAS);
    }
    tables_ready = 1;
}

float e2m1_value(uint8_t code) { return E2M1_VALUES[code & CODE_MASK]; }

float e8m0_scale(uint8_t biased_exponent) {
    prepare_tables();
    return e8m0_scales[biased_exponent];
}

uint32_t next_random(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static float unit_random(uint32_t *state) {
    return (float)(next_random(state) >> 8) / (float)(1 << 24) * 2.0f - 1.0f;
}

/* ---- 32-bit floats: 4 bytes per weight, no scale ---- */

static float dot_fp32(const uint8_t *elements, const uint8_t *scales, const float *activations, size_t blocks) {
    (void)scales;
    const float *weights = (const float *)elements;
    float total = 0.0f;
    for (size_t i = 0; i < blocks * WEIGHTS_PER_BLOCK; i++) total += weights[i] * activations[i];
    return total;
}

static void fill_fp32(uint8_t *elements, uint8_t *scales, size_t blocks, uint32_t seed) {
    (void)scales;
    float *weights = (float *)elements;
    for (size_t i = 0; i < blocks * WEIGHTS_PER_BLOCK; i++) weights[i] = unit_random(&seed);
}

/* ---- 8-bit integers with one 32-bit float scale per block ---- */

static float dot_int8(const uint8_t *elements, const uint8_t *scales, const float *activations, size_t blocks) {
    const int8_t *weights = (const int8_t *)elements;
    const float *block_scales = (const float *)scales;
    float total = 0.0f;
    for (size_t block = 0; block < blocks; block++) {
        float block_sum = 0.0f;
        for (size_t i = 0; i < WEIGHTS_PER_BLOCK; i++) block_sum += (float)weights[i] * activations[i];
        total += block_sum * block_scales[block];
        weights += WEIGHTS_PER_BLOCK;
        activations += WEIGHTS_PER_BLOCK;
    }
    return total;
}

static void fill_int8(uint8_t *elements, uint8_t *scales, size_t blocks, uint32_t seed) {
    float *block_scales = (float *)scales;
    for (size_t i = 0; i < blocks * WEIGHTS_PER_BLOCK; i++) elements[i] = (uint8_t)next_random(&seed);
    for (size_t block = 0; block < blocks; block++) block_scales[block] = INT8_FILL_SCALE;
}

/* ---- MXFP4: two 4-bit codes per byte, one E8M0 scale byte per block ---- */

static float dot_mxfp4(const uint8_t *elements, const uint8_t *scales, const float *activations, size_t blocks) {
    prepare_tables();
    float total = 0.0f;
    for (size_t block = 0; block < blocks; block++) {
        float block_sum = 0.0f;
        for (size_t i = 0; i < WEIGHTS_PER_BLOCK; i++) {
            uint8_t code = (elements[i / CODES_PER_BYTE] >> (CODE_BITS * (i % CODES_PER_BYTE))) & CODE_MASK;
            block_sum += E2M1_VALUES[code] * activations[i];
        }
        total += block_sum * e8m0_scales[scales[block]];
        elements += WEIGHTS_PER_BLOCK / CODES_PER_BYTE;
        activations += WEIGHTS_PER_BLOCK;
    }
    return total;
}

/* The same format, decoding both codes of a byte with one lookup in a 256-entry table. */
static float dot_mxfp4_pair_table(const uint8_t *elements, const uint8_t *scales, const float *activations,
                                  size_t blocks) {
    prepare_tables();
    float total = 0.0f;
    for (size_t block = 0; block < blocks; block++) {
        float block_sum = 0.0f;
        for (size_t byte = 0; byte < WEIGHTS_PER_BLOCK / CODES_PER_BYTE; byte++) {
            CodePair pair = code_pairs[elements[byte]];
            block_sum += pair.low * activations[CODES_PER_BYTE * byte]
                       + pair.high * activations[CODES_PER_BYTE * byte + 1];
        }
        total += block_sum * e8m0_scales[scales[block]];
        elements += WEIGHTS_PER_BLOCK / CODES_PER_BYTE;
        activations += WEIGHTS_PER_BLOCK;
    }
    return total;
}

static void fill_mxfp4(uint8_t *elements, uint8_t *scales, size_t blocks, uint32_t seed) {
    for (size_t i = 0; i < blocks * WEIGHTS_PER_BLOCK / CODES_PER_BYTE; i++) elements[i] = (uint8_t)next_random(&seed);
    for (size_t block = 0; block < blocks; block++)
        scales[block] = (uint8_t)(SMALLEST_FILLED_EXPONENT + next_random(&seed) % FILLED_EXPONENT_RANGE);
}

const WeightFormat WEIGHT_FORMATS[] = {
    {"fp32", WEIGHTS_PER_BLOCK * sizeof(float), 0, dot_fp32, fill_fp32},
    {"int8_fp32_scale", WEIGHTS_PER_BLOCK, sizeof(float), dot_int8, fill_int8},
    {"mxfp4", WEIGHTS_PER_BLOCK / CODES_PER_BYTE, 1, dot_mxfp4, fill_mxfp4},
    {"mxfp4_pair_table", WEIGHTS_PER_BLOCK / CODES_PER_BYTE, 1, dot_mxfp4_pair_table, fill_mxfp4},
};
const size_t WEIGHT_FORMAT_COUNT = sizeof(WEIGHT_FORMATS) / sizeof(WEIGHT_FORMATS[0]);

double bytes_per_weight(const WeightFormat *format) {
    return (double)(format->element_bytes_per_block + format->scale_bytes_per_block) / WEIGHTS_PER_BLOCK;
}

float decoded_weight(const WeightFormat *format, const uint8_t *elements, const uint8_t *scales, size_t index) {
    if (format->dot == dot_fp32) return ((const float *)elements)[index];
    if (format->dot == dot_int8) return (float)((const int8_t *)elements)[index] * ((const float *)scales)[0];
    uint8_t code = elements[index / CODES_PER_BYTE] >> (CODE_BITS * (index % CODES_PER_BYTE));
    return e2m1_value(code) * e8m0_scale(scales[0]);
}
