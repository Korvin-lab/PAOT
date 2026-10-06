#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define NC 50
#define SEG 18
#define HC (1u << 18)
#ifndef EXPECTED_INPUT
#define EXPECTED_INPUT UINT64_C(138583872)
#endif
typedef struct {
    double segment, sum[NC], correction[NC];
    unsigned char count[NC], days;
} Entry;
typedef struct { uint64_t count, missing; long double sum, correction; double min, max; } Stats;
static Entry *entries;
static uint32_t *table, used, entry_capacity;
static Stats raw_stats[NC], weekly_stats[NC];
static long double recovered[NC], recovered_correction[NC];
static uint64_t input_rows, output_rows, group_input, group_days, groups;
static char names[NC][100], group_id[32], group_week[11];
static FILE *destination, *coverage;
static CC_SHA256_CTX output_hash;

static void fail(const char *s) { fprintf(stderr, "FAIL %s\n", s); exit(1); }
static int numerical(int col) { return col >= 2 && col != SEG; }
static int fields(char *line, char *out[NC]) {
    int col = 0; out[0] = line;
    for (char *p = line; *p; p++) {
        if (*p == '"') fail("quoted data");
        if (*p == ',') { *p = 0; if (++col >= NC) fail("too many fields"); out[col] = p + 1; }
        else if (*p == '\r' || *p == '\n') { *p = 0; break; }
    }
    if (col != NC - 1) fail("field count");
    return col + 1;
}
static double number(char *text) {
    char *end; double value = strtod(text, &end);
    if (!*text || *end || !isfinite(value)) fail("invalid numeric");
    return value;
}
static Entry *get_entry(double segment) {
    if (segment == 0) segment = 0;
    uint64_t bits; memcpy(&bits, &segment, sizeof bits);
    bits ^= bits >> 33; bits *= UINT64_C(0xff51afd7ed558ccd); bits ^= bits >> 33;
    uint32_t slot = (uint32_t)bits & (HC - 1);
    while (table[slot]) {
        Entry *entry = &entries[table[slot] - 1];
        if (entry->segment == segment) return entry;
        slot = (slot + 1) & (HC - 1);
    }
    if (used >= HC / 2) fail("segment table capacity");
    if (used == entry_capacity) {
        entry_capacity = entry_capacity ? entry_capacity * 2 : 1024;
        entries = realloc(entries, entry_capacity * sizeof(*entries));
        if (!entries) fail("segment allocation");
    }
    Entry *entry = &entries[used]; memset(entry, 0, sizeof *entry); entry->segment = segment;
    table[slot] = ++used; return entry;
}
static void add_stats(Stats *stats, double value) {
    if (!stats->count || value < stats->min) stats->min = value;
    if (!stats->count || value > stats->max) stats->max = value;
    stats->count++;
    long double y = (long double)value - stats->correction;
    long double updated = stats->sum + y;
    stats->correction = (updated - stats->sum) - y;
    stats->sum = updated;
}
static int compare_entries(const void *a, const void *b) {
    double x = ((const Entry *)a)->segment, y = ((const Entry *)b)->segment;
    return (x > y) - (x < y);
}
static void write_bytes(const char *text, size_t size) {
    if (fwrite(text, 1, size, destination) != size) fail("output write");
    CC_SHA256_Update(&output_hash, text, (CC_LONG)size);
}
static void flush_group(void) {
    if (!used) return;
    qsort(entries, used, sizeof(*entries), compare_entries);
    unsigned min_days = 8, max_days = 0;
    uint64_t kv_rows = 0, ing_rows = 0;
    for (uint32_t i = 0; i < used; i++) {
        Entry *entry = &entries[i]; unsigned observations = (unsigned)__builtin_popcount((unsigned)entry->days);
        if (observations < min_days) min_days = observations;
        if (observations > max_days) max_days = observations;
        char output[8192]; size_t length = 0;
        for (int col = 0; col < NC; col++) {
            if (col) output[length++] = ',';
            if (col == 0) length += (size_t)snprintf(output + length, sizeof output - length, "%s", group_week);
            else if (col == 1) length += (size_t)snprintf(output + length, sizeof output - length, "%s", group_id);
            else if (col == SEG) length += (size_t)snprintf(output + length, sizeof output - length, "%.17g", entry->segment);
            else if (entry->count[col]) {
                if (entry->count[col] > observations) fail("observation count");
                double mean = entry->sum[col] / entry->count[col];
                add_stats(&weekly_stats[col], mean);
                long double y = (long double)mean * entry->count[col] - recovered_correction[col];
                long double updated = recovered[col] + y;
                recovered_correction[col] = (updated - recovered[col]) - y;
                recovered[col] = updated;
                length += (size_t)snprintf(output + length, sizeof output - length, "%.17g", mean);
            } else weekly_stats[col].missing++;
            if (length >= sizeof output - 100) fail("output buffer");
        }
        output[length++] = '\n'; write_bytes(output, length); output_rows++;
        kv_rows += entry->count[7] != 0; ing_rows += entry->count[11] != 0;
    }
    fprintf(coverage, "%s,%s,%u,%u,%" PRIu64 ",%u,%u,%" PRIu64 ",%" PRIu64 "\n",
            group_id, group_week, (unsigned)__builtin_popcount((unsigned)group_days), used,
            group_input, min_days, max_days, kv_rows, ing_rows);
    groups++;
    if (groups % 250 == 0) fprintf(stderr, "weeks=%" PRIu64 " input=%" PRIu64 " output=%" PRIu64 "\n", groups, input_rows, output_rows);
    used = 0; group_input = 0; group_days = 0; memset(table, 0, HC * sizeof(*table));
}

int main(int argc, char **argv) {
    if (argc != 6) fail("usage SOURCE PLAN OUTPUT COVERAGE REPORT");
    FILE *source = fopen(argv[1], "rb"), *plan = fopen(argv[2], "rb");
    destination = fopen(argv[3], "wb"); coverage = fopen(argv[4], "wb");
    if (!source || !plan || !destination || !coverage) fail("open");
    setvbuf(source, NULL, _IOFBF, 16 * 1024 * 1024);
    setvbuf(destination, NULL, _IOFBF, 16 * 1024 * 1024);
    table = calloc(HC, sizeof(*table)); if (!table) fail("hash allocation");
    CC_SHA256_Init(&output_hash);
    char *line = NULL; size_t capacity = 0; ssize_t length = getline(&line, &capacity, source);
    if (length < 1) fail("source header");
    write_bytes(line, (size_t)length);
    char *parts[NC]; fields(line, parts);
    for (int col = 0; col < NC; col++) {
        if (strlen(parts[col]) >= sizeof names[col]) fail("column name");
        strcpy(names[col], parts[col]);
    }
    if (strcmp(names[SEG], "segment_id") || strcmp(names[7], "kvch") || strcmp(names[11], "ing_factor")) fail("schema");
    fputs("id,week,dates,segments,input_rows,min_days_per_segment,max_days_per_segment,kvch_rows,ing_rows\n", coverage);
    char *plan_line = NULL; size_t plan_capacity = 0;
    if (getline(&plan_line, &plan_capacity, plan) < 1) fail("plan header");
    while (getline(&plan_line, &plan_capacity, plan) > 0) {
        char id[32], week[11], day[11]; unsigned day_index;
        uint64_t offset, bytes, rows;
        if (sscanf(plan_line, "%31[^,],%10[^,],%10[^,],%u,%" SCNu64 ",%" SCNu64 ",%" SCNu64,
                   id, week, day, &day_index, &offset, &bytes, &rows) != 7 || day_index > 6) fail("plan record");
        if (strcmp(id, group_id) || strcmp(week, group_week)) {
            flush_group(); strcpy(group_id, id); strcpy(group_week, week);
        }
        if (fseeko(source, (off_t)offset, SEEK_SET)) fail("source seek");
        uint64_t read_bytes = 0;
        for (uint64_t row = 0; row < rows; row++) {
            length = getline(&line, &capacity, source); if (length < 1) fail("source block eof");
            read_bytes += (uint64_t)length;
            fields(line, parts);
            const char *date_text = parts[0]; if ((unsigned char)date_text[0] == 0xef) date_text += 3;
            if (strcmp(date_text, day) || strcmp(parts[1], id)) fail("source block key");
            Entry *entry = get_entry(number(parts[SEG]));
            unsigned char bit = (unsigned char)(1u << day_index);
            if (entry->days & bit) fail("duplicate pipe day segment");
            entry->days |= bit;
            for (int col = 2; col < NC; col++) {
                if (!numerical(col)) continue;
                if (!*parts[col]) { raw_stats[col].missing++; continue; }
                double value = number(parts[col]);
                double y = value - entry->correction[col];
                double updated = entry->sum[col] + y;
                entry->correction[col] = (updated - entry->sum[col]) - y;
                entry->sum[col] = updated; entry->count[col]++;
                add_stats(&raw_stats[col], value);
            }
            input_rows++; group_input++; group_days |= bit;
        }
        if (read_bytes != bytes) fail("block byte count");
    }
    flush_group();
    if (ferror(plan) || input_rows != EXPECTED_INPUT) fail("input row total");
    for (int col = 2; col < NC; col++) if (numerical(col)) {
        if (raw_stats[col].count + raw_stats[col].missing != input_rows ||
            weekly_stats[col].count + weekly_stats[col].missing != output_rows) fail("count conservation");
        long double tolerance = 1e-12L * (fabsl(raw_stats[col].sum) + 1);
        if (fabsl(recovered[col] - raw_stats[col].sum) > tolerance) fail("weighted mean conservation");
    }
    if (fflush(destination) || fsync(fileno(destination)) || fclose(destination) || fclose(source) || fclose(plan) || fclose(coverage)) fail("close");
    unsigned char digest[32]; char hex[65]; CC_SHA256_Final(digest, &output_hash);
    for (int i = 0; i < 32; i++) sprintf(hex + 2*i, "%02x", digest[i]);
    FILE *report = fopen(argv[5], "wb"); if (!report) fail("report open");
    fprintf(report, "{\n\"input_rows\":%" PRIu64 ",\"output_rows\":%" PRIu64 ",\"pipe_weeks\":%" PRIu64 ",\"sha256\":\"%s\",\"columns\":[\n",
            input_rows, output_rows, groups, hex);
    int first = 1;
    for (int col = 2; col < NC; col++) if (numerical(col)) {
        if (!first) fputs(",\n", report); first = 0;
        fprintf(report, "{\"name\":\"%s\",\"source_count\":%" PRIu64 ",\"source_missing\":%" PRIu64 ",\"weekly_count\":%" PRIu64 ",\"weekly_missing\":%" PRIu64 ",\"source_min\":%.17g,\"source_max\":%.17g}",
                names[col], raw_stats[col].count, raw_stats[col].missing,
                weekly_stats[col].count, weekly_stats[col].missing, raw_stats[col].min, raw_stats[col].max);
    }
    fputs("\n]}\n", report); if (fclose(report)) fail("report close");
    printf("PASS input=%" PRIu64 " weekly_rows=%" PRIu64 " pipe_weeks=%" PRIu64 " output_sha256=%s\n", input_rows, output_rows, groups, hex);
    free(entries); free(table); free(line); free(plan_line); return 0;
}
