#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CAP (1u << 18)
#ifndef ROWS
#define ROWS UINT64_C(138583872)
#endif
#ifndef KEYS
#define KEYS 71460
#endif
typedef struct { uint64_t id; uint32_t date,hits; char *value; } Entry;
static Entry *table;
static void fail(const char *msg) { fprintf(stderr,"FAIL: %s\n",msg);exit(1); }
static uint64_t idnum(const char *a,const char *b) {
 uint64_t v=0;if(a==b)fail("empty id");
 for(;a<b;a++){if(*a<'0'||*a>'9')fail("invalid id");v=v*10+(unsigned)(*a-'0');}return v;
}
static uint32_t daten(const char *a,const char *b) {
 if(b-a!=10||a[4]!='-'||a[7]!='-')fail("invalid date");
 uint32_t v=0;for(int i=0;i<10;i++)if(i!=4&&i!=7){if(a[i]<'0'||a[i]>'9')fail("invalid date digit");v=v*10+(unsigned)(a[i]-'0');}return v;
}
static Entry *find(uint64_t id,uint32_t date) {
 uint64_t h=id*UINT64_C(11400714819323198485)^(uint64_t)date*UINT64_C(14029467366897019727);
 uint32_t i=(uint32_t)(h^(h>>32))&(CAP-1);
 for(;;){Entry *e=&table[i];if(!e->id||(e->id==id&&e->date==date))return e;i=(i+1)&(CAP-1);}
}
static void hash_write(FILE *out,CC_SHA256_CTX *hash,const char *s,size_t n) {
 if(n&&fwrite(s,1,n,out)!=n)fail("write failed");
 if(n&&!CC_SHA256_Update(hash,s,(CC_LONG)n))fail("output hash failed");
}
static void hexhash(CC_SHA256_CTX *ctx,char hex[65]) {
 unsigned char v[32];const char *d="0123456789abcdef";CC_SHA256_Final(v,ctx);
 for(int i=0;i<32;i++){hex[i*2]=d[v[i]>>4];hex[i*2+1]=d[v[i]&15];}hex[64]=0;
}
int main(int argc,char **argv) {
 if(argc!=5)fail("usage: patch_min PATCH.csv SOURCE.csv TEMP.csv EXPECTED_SOURCE_SHA256");
 table=calloc(CAP,sizeof(Entry));if(!table)fail("allocation");
 FILE *patch=fopen(argv[1],"rb");if(!patch)fail("patch open");
 char *line=NULL;size_t cap=0;ssize_t n=getline(&line,&cap,patch);
 if(n<=0||strcmp(line,"id,date,Min\n"))fail("patch header");
 uint32_t keys=0;
 while((n=getline(&line,&cap,patch))>0){
  char *v[3],*save=NULL;int m=0;
  for(char *p=strtok_r(line,",\r\n",&save);p&&m<3;p=strtok_r(NULL,",\r\n",&save))v[m++]=p;
  if(m!=3)fail("patch row malformed");
  char *end;double x=strtod(v[2],&end);if(*end||!isfinite(x)||x<10||x>60)fail("patch Min out of range");
  Entry *e=find(idnum(v[0],v[0]+strlen(v[0])),daten(v[1],v[1]+strlen(v[1])));
  if(e->id)fail("duplicate patch key");e->id=idnum(v[0],v[0]+strlen(v[0]));e->date=daten(v[1],v[1]+strlen(v[1]));
  if(!(e->value=strdup(v[2])))fail("allocation");keys++;
 }
 if(ferror(patch)||fclose(patch)||keys!=KEYS)fail("patch key count");
 FILE *src=fopen(argv[2],"rb"),*out=fopen(argv[3],"wb");if(!src||!out)fail("CSV open");
 setvbuf(src,NULL,_IOFBF,16*1024*1024);setvbuf(out,NULL,_IOFBF,16*1024*1024);
 CC_SHA256_CTX sh,oh;CC_SHA256_Init(&sh);CC_SHA256_Init(&oh);
 n=getline(&line,&cap,src);if(n<=0||strncmp(line,"\xef\xbb\xbf" "date,id,",11))fail("source header");
 CC_SHA256_Update(&sh,line,(CC_LONG)n);hash_write(out,&oh,line,(size_t)n);
 uint64_t rows=0,patched=0,salinity_missing=0,salinity_old_min_ppm=0;
 while((n=getline(&line,&cap,src))>0){
  if(line[n-1]!='\n'||memchr(line,'"',(size_t)n))fail("unexpected CSV row");
  CC_SHA256_Update(&sh,line,(CC_LONG)n);
  const char *comma=memchr(line,',',(size_t)n),*next=comma?memchr(comma+1,',',(size_t)(line+n-comma-1)):NULL;
  if(!next)fail("row key");Entry *e=find(idnum(comma+1,next),daten(line,comma));
  if(!e->id)hash_write(out,&oh,line,(size_t)n);
  else {
   const char *start=line,*minstart=NULL,*minend=NULL,*salstart=NULL,*salend=NULL;int col=0;
   for(const char *p=line;p<line+n;p++)if(*p==','){
    if(col==9)minend=p;
    if(col==26)salend=p;
    col++;if(col==9)minstart=p+1;
    if(col==26)salstart=p+1;
   }
   if(col!=50||!minstart||!minend||!salstart||!salend)fail("column count");
   if(salstart==salend)salinity_missing++;
   else {
    char *end;double sal=strtod(salstart,&end);
    if(end!=salend||!isfinite(sal))fail("invalid seg_salinity");
    if(fabs(sal-22662.7)<1e-6)salinity_old_min_ppm++;
   }
   hash_write(out,&oh,start,(size_t)(minstart-start));
   hash_write(out,&oh,e->value,strlen(e->value));
   hash_write(out,&oh,minend,(size_t)(line+n-minend));
   e->hits++;patched++;
  }
  if(++rows%UINT64_C(10000000)==0)fprintf(stderr,"processed %" PRIu64 " rows; patched %" PRIu64 "\n",rows,patched);
 }
 if(ferror(src)||rows!=ROWS)fail("source read/row count");
 uint32_t seen=0;for(uint32_t i=0;i<CAP;i++)if(table[i].id&&table[i].hits)seen++;
 if(seen!=keys)fail("not all keys present in CSV");
 if(fflush(out)||fsync(fileno(out))||fclose(out)||fclose(src))fail("flush/close");
 char source_hex[65],output_hex[65];hexhash(&sh,source_hex);hexhash(&oh,output_hex);
 if(strcmp(source_hex,argv[4]))fail("source SHA mismatch; temp retained");
 printf("PASS rows=%" PRIu64 " patched_segment_rows=%" PRIu64 " matched_keys=%u seg_salinity_missing_in_patched_rows=%" PRIu64 " seg_salinity_equal_old_min_ppm_in_patched_rows=%" PRIu64 " source_sha256=%s output_sha256=%s\n",rows,patched,seen,salinity_missing,salinity_old_min_ppm,source_hex,output_hex);
 free(line);free(table);return 0;
}
