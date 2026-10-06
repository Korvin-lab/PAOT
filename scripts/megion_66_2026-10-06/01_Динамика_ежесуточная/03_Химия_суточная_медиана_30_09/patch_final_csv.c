#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CAP (1u << 18)
#ifndef ROWS
#define ROWS UINT64_C(138583872)
#endif
#ifndef PATCH_KEYS
#define PATCH_KEYS 88721
#endif
typedef struct { uint64_t id; uint32_t day, hits; char *v[4]; } Entry;
static Entry *table;

static void fail(const char *s) { fprintf(stderr, "ERROR: %s\n", s); exit(1); }
static uint64_t number(const char *a, const char *b) {
    uint64_t v=0; if(a==b) fail("empty ID");
    for(;a<b;a++) { if(*a<'0'||*a>'9') fail("bad ID"); v=v*10+(unsigned)(*a-'0'); }
    return v;
}
static uint32_t daykey(const char *a,const char *b) {
    if(b-a!=10||a[4]!='-'||a[7]!='-') fail("bad date");
    uint32_t v=0;
    for(int i=0;i<10;i++) if(i!=4&&i!=7) {
        if(a[i]<'0'||a[i]>'9') fail("bad date digit");
        v=v*10+(unsigned)(a[i]-'0');
    }
    return v;
}
static Entry *lookup(uint64_t id,uint32_t day) {
    uint64_t h=id*UINT64_C(11400714819323198485)^(uint64_t)day*UINT64_C(14029467366897019727);
    uint32_t i=(uint32_t)(h^(h>>32))&(CAP-1);
    for(;;) { Entry *e=&table[i]; if(!e->id||(e->id==id&&e->day==day)) return e; i=(i+1)&(CAP-1); }
}
static void write_hash(FILE *f,CC_SHA256_CTX *hash,const char *s,size_t n) {
    if(n&&fwrite(s,1,n,f)!=n) fail("write failed");
    if(n&&!CC_SHA256_Update(hash,s,(CC_LONG)n)) fail("hash failed");
}
static void hexhash(CC_SHA256_CTX *ctx,char out[65]) {
    unsigned char d[32]; const char *hex="0123456789abcdef";
    CC_SHA256_Final(d,ctx);
    for(int i=0;i<32;i++){out[2*i]=hex[d[i]>>4];out[2*i+1]=hex[d[i]&15];}
    out[64]=0;
}
int main(int argc,char **argv) {
    if(argc!=5) fail("usage: patch PATCH.csv SOURCE.csv OUTPUT.partial.csv EXPECTED_SOURCE_SHA256");
    table=calloc(CAP,sizeof(Entry)); if(!table) fail("allocation failed");
    FILE *patch=fopen(argv[1],"rb"); if(!patch) fail("patch file missing");
    char *line=NULL; size_t cap=0; ssize_t n=getline(&line,&cap,patch);
    if(n<=0||strcmp(line,"id,date,CO2,pH,H2S in Gas Phase,pCO2\n")) fail("patch header mismatch");
    uint32_t keys=0;
    while((n=getline(&line,&cap,patch))>0) {
        char *fields[6],*save=NULL; int count=0;
        for(char *p=strtok_r(line,",\r\n",&save);p&&count<6;p=strtok_r(NULL,",\r\n",&save)) fields[count++]=p;
        if(count!=6) fail("patch row malformed");
        uint64_t id=number(fields[0],fields[0]+strlen(fields[0]));
        uint32_t day=daykey(fields[1],fields[1]+strlen(fields[1]));
        Entry *e=lookup(id,day); if(e->id) fail("duplicate patch key");
        e->id=id;e->day=day;
        for(int j=0;j<4;j++) if(!(e->v[j]=strdup(fields[j+2]))) fail("allocation failed");
        keys++;
    }
    if(ferror(patch)||fclose(patch)||keys!=PATCH_KEYS) fail("patch count/reading failed");
    FILE *src=fopen(argv[2],"rb"),*out=fopen(argv[3],"wb");
    if(!src||!out) fail("source or output open failed");
    setvbuf(src,NULL,_IOFBF,16*1024*1024);setvbuf(out,NULL,_IOFBF,16*1024*1024);
    CC_SHA256_CTX src_hash,out_hash; CC_SHA256_Init(&src_hash);CC_SHA256_Init(&out_hash);
    n=getline(&line,&cap,src);
    if(n<=0||strncmp(line,"\xef\xbb\xbf" "date,id,",11)) fail("source header mismatch");
    CC_SHA256_Update(&src_hash,line,(CC_LONG)n);write_hash(out,&out_hash,line,(size_t)n);
    uint64_t rows=0,patched=0;
    while((n=getline(&line,&cap,src))>0) {
        if(line[n-1]!='\n'||memchr(line,'"',(size_t)n)) fail("unexpected CSV row");
        CC_SHA256_Update(&src_hash,line,(CC_LONG)n);
        const char *a=memchr(line,',',(size_t)n),*b=a?memchr(a+1,',',(size_t)(line+n-a-1)):NULL;
        if(!b) fail("missing CSV keys");
        Entry *e=lookup(number(a+1,b),daykey(line,a));
        if(!e->id) write_hash(out,&out_hash,line,(size_t)n);
        else {
            const char *start[51],*end[51];int col=0;start[0]=line;
            for(const char *p=line;p<line+n;p++) if(*p==',') {
                if(col==50) fail("too many columns");
                end[col]=p;start[++col]=p+1;
            }
            if(col!=50) fail("column count mismatch");
            end[50]=line+n-1;
            if(end[50]>start[50]&&end[50][-1]=='\r') end[50]--;
            write_hash(out,&out_hash,line,(size_t)(start[8]-line));
            write_hash(out,&out_hash,e->v[0],strlen(e->v[0]));
            write_hash(out,&out_hash,end[8],(size_t)(start[10]-end[8]));
            write_hash(out,&out_hash,e->v[1],strlen(e->v[1]));
            write_hash(out,&out_hash,end[10],(size_t)(start[38]-end[10]));
            write_hash(out,&out_hash,e->v[3],strlen(e->v[3]));
            write_hash(out,&out_hash,end[38],(size_t)(start[50]-end[38]));
            write_hash(out,&out_hash,e->v[2],strlen(e->v[2]));
            write_hash(out,&out_hash,end[50],(size_t)(line+n-end[50]));
            e->hits++;patched++;
        }
        if(++rows%UINT64_C(10000000)==0) fprintf(stderr,"processed %" PRIu64 " rows; changed %" PRIu64 "\n",rows,patched);
    }
    if(ferror(src)||rows!=ROWS) fail("source read/row count mismatch");
    uint32_t seen=0;for(uint32_t i=0;i<CAP;i++) if(table[i].id&&table[i].hits) seen++;
    if(seen!=keys) fail("not all patch keys were present");
    if(fflush(out)||fsync(fileno(out))||fclose(out)||fclose(src)) fail("flush/close failed");
    char sh1[65],sh2[65];hexhash(&src_hash,sh1);hexhash(&out_hash,sh2);
    if(strcmp(sh1,argv[4])) fail("original SHA-256 mismatch; partial output retained");
    printf("rows=%" PRIu64 "\npatched_segment_rows=%" PRIu64 "\nmatched_id_dates=%u\nsource_sha256=%s\noutput_sha256=%s\n",rows,patched,seen,sh1,sh2);
    free(line);free(table);
    return 0;
}
