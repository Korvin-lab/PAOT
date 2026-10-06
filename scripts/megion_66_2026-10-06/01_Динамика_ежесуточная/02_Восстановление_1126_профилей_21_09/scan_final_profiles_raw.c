#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Scan an unquoted CSV in fixed blocks. This deliberately does not use getline:
 * one anomalously long source line must be reported, not exhaust process memory.
 */
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s INPUT.csv OUTPUT.tsv\n", argv[0]); return 2; }
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) { perror("open"); return 2; }
    const size_t block_size = 8 * 1024 * 1024;
    unsigned char *block = malloc(block_size);
    if (!block) { perror("malloc"); return 2; }
    char key[160] = {0}, previous[160] = {0};
    size_t key_len = 0;
    unsigned long long rows = 0, profiles = 0, bytes = 0;
    int commas = 0, header = 1;
    size_t count;
    while ((count = fread(block, 1, block_size, in)) > 0) {
        for (size_t i = 0; i < count; ++i) {
            unsigned char c = block[i]; bytes++;
            if (c == '\n') {
                if (header) {
                    header = 0;
                } else {
                    rows++;
                    if (commas != 50) {
                        fprintf(stderr, "malformed row %llu at byte %llu: %d commas\n", rows + 1, bytes, commas);
                        return 1;
                    }
                    if (key_len == 0 || key_len >= sizeof(key)) {
                        fprintf(stderr, "missing/oversize key at row %llu\n", rows + 1); return 1;
                    }
                    key[key_len] = '\0';
                    if (strcmp(key, previous) != 0) {
                        char *sep = strchr(key, ',');
                        if (!sep) { fprintf(stderr, "invalid key at row %llu\n", rows + 1); return 1; }
                        *sep = '\0';
                        fprintf(out, "%s\t%s\n", sep + 1, key);
                        *sep = ',';
                        strcpy(previous, key);
                        profiles++;
                    }
                }
                commas = 0; key_len = 0;
                continue;
            }
            if (c == ',') commas++;
            if (!header && commas <= 1 && key_len + 1 < sizeof(key)) key[key_len++] = (char)c;
        }
    }
    if (ferror(in)) { perror("read"); return 2; }
    free(block); fclose(in); fclose(out);
    printf("rows=%llu profiles=%llu bytes=%llu\n", rows, profiles, bytes);
    return 0;
}
