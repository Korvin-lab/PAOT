#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail(const char *message) { fprintf(stderr, "FAIL %s\n", message); exit(1); }
static void emit(FILE *out, const char *id, const char *day, uint64_t offset,
                 uint64_t bytes, uint64_t rows) {
    if (fprintf(out, "%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                id, day, offset, bytes, rows) < 0) fail("index write");
}
int main(int argc, char **argv) {
    if (argc != 4) fail("usage SOURCE INDEX EXPECTED_SHA");
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
    if (!in || !out) fail("open");
    setvbuf(in, NULL, _IOFBF, 16 * 1024 * 1024);
    CC_SHA256_CTX hash; CC_SHA256_Init(&hash);
    char *line = NULL; size_t capacity = 0; ssize_t length;
    uint64_t cursor = 0, rows = 0, blocks = 0, offset = 0, bytes = 0, block_rows = 0;
    char last_id[32] = "", last_day[11] = "";
    length = getline(&line, &capacity, in);
    if (length < 1 || strncmp(line, "\xef\xbb\xbf" "date,id,", 11)) fail("header");
    CC_SHA256_Update(&hash, line, (CC_LONG)length); cursor = (uint64_t)length;
    fputs("id,date,offset,bytes,rows\n", out);
    while ((length = getline(&line, &capacity, in)) > 0) {
        CC_SHA256_Update(&hash, line, (CC_LONG)length);
        char *first = strchr(line, ','), *second = first ? strchr(first + 1, ',') : NULL;
        if (!second || first - line != 10 || second - first - 1 > 30) fail("key");
        char day[11], id[32]; memcpy(day, line, 10); day[10] = 0;
        size_t n = (size_t)(second - first - 1); memcpy(id, first + 1, n); id[n] = 0;
        if (strcmp(id, last_id) || strcmp(day, last_day)) {
            if (block_rows) { emit(out, last_id, last_day, offset, bytes, block_rows); blocks++; }
            strcpy(last_id, id); strcpy(last_day, day); offset = cursor; bytes = 0; block_rows = 0;
        }
        bytes += (uint64_t)length; block_rows++; rows++; cursor += (uint64_t)length;
        if (rows % UINT64_C(10000000) == 0) fprintf(stderr, "indexed %" PRIu64 " rows\n", rows);
    }
    if (block_rows) { emit(out, last_id, last_day, offset, bytes, block_rows); blocks++; }
    if (ferror(in) || rows != UINT64_C(138583872)) fail("source rows");
    unsigned char digest[32]; char hex[65]; CC_SHA256_Final(digest, &hash);
    for (int i = 0; i < 32; i++) sprintf(hex + 2 * i, "%02x", digest[i]);
    if (strcmp(hex, argv[3])) fail("source SHA");
    if (fflush(out) || fsync(fileno(out)) || fclose(out) || fclose(in)) fail("close");
    printf("PASS rows=%" PRIu64 " blocks=%" PRIu64 " bytes=%" PRIu64 " source_sha256=%s\n",
           rows, blocks, cursor, hex);
    free(line); return 0;
}
