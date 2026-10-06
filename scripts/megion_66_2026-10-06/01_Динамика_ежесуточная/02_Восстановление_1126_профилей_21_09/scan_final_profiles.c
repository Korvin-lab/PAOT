#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Extract one line per contiguous date/id profile from an unquoted final CSV. */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT.csv OUTPUT.tsv\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) { perror("open"); return 2; }
    static char inbuf[8 * 1024 * 1024];
    static char outbuf[1024 * 1024];
    setvbuf(in, inbuf, _IOFBF, sizeof(inbuf));
    setvbuf(out, outbuf, _IOFBF, sizeof(outbuf));

    char *line = NULL, *previous = NULL;
    size_t cap = 0, previous_cap = 0;
    ssize_t n;
    unsigned long long rows = 0, profiles = 0;
    if ((n = getline(&line, &cap, in)) < 0) { fprintf(stderr, "empty file\n"); return 1; }
    if (strncmp(line, "\xEF\xBB\xBF" "date,id,Qv,", 14) != 0 && strncmp(line, "date,id,Qv,", 11) != 0) {
        fprintf(stderr, "unexpected header\n"); return 1;
    }
    while ((n = getline(&line, &cap, in)) >= 0) {
        rows++;
        int commas = 0;
        char *first = NULL, *second = NULL;
        for (ssize_t i = 0; i < n; ++i) {
            if (line[i] == ',') {
                commas++;
                if (!first) first = line + i;
                else if (!second) second = line + i;
            }
        }
        if (commas != 50 || !first || !second) {
            fprintf(stderr, "malformed row %llu: %d commas\n", rows + 1, commas);
            return 1;
        }
        size_t key_len = (size_t)(second - line);
        if (!previous || strlen(previous) != key_len || memcmp(previous, line, key_len) != 0) {
            if (key_len + 1 > previous_cap) {
                previous_cap = key_len + 64;
                previous = realloc(previous, previous_cap);
                if (!previous) { perror("realloc"); return 2; }
            }
            memcpy(previous, line, key_len);
            previous[key_len] = '\0';
            /* Convert date,id to id<TAB>date for later Python lookup. */
            fwrite(first + 1, 1, (size_t)(second - first - 1), out);
            fputc('\t', out);
            fwrite(line, 1, (size_t)(first - line), out);
            fputc('\n', out);
            profiles++;
        }
    }
    free(line); free(previous);
    fclose(in); fclose(out);
    fprintf(stdout, "rows=%llu profiles=%llu\n", rows, profiles);
    return 0;
}
