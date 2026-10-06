#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CAP (1u << 18)
#ifndef EXPECTED_ROWS
#define EXPECTED_ROWS UINT64_C(138583872)
#endif
#ifndef EXPECTED_KEYS
#define EXPECTED_KEYS 63276
#endif
typedef struct { uint64_t id; uint32_t day,hits; char *values[4]; } Row;
static Row *table;
static void fail(const char *msg) { fprintf(stderr,"VERIFY FAIL: %s\n",msg);exit(1); }
static uint64_t idnum(const char *a,const char *b) {
    uint64_t v=0;if(a==b)fail("empty ID");
    while(a<b){if(*a<'0'||*a>'9')fail("bad ID");v=v*10+(unsigned)(*a++-'0');}return v;
}
static uint32_t daynum(const char *a,const char *b) {
    if(b-a!=10||a[4]!='-'||a[7]!='-')fail("bad day");
    uint32_t v=0;
    for(int i=0;i<10;i++)if(i!=4&&i!=7){if(a[i]<'0'||a[i]>'9')fail("bad day digit");v=v*10+(unsigned)(a[i]-'0');}
    return v;
}
static Row *get(uint64_t id,uint32_t day) {
    uint64_t h=(id<<17)^((uint64_t)day*UINT64_C(0x9e3779b97f4a7c15));
    uint32_t i=(uint32_t)(h^(h>>29))&(CAP-1);
    for(;;){Row *r=&table[i];if(!r->id||(r->id==id&&r->day==day))return r;i=(i+1)&(CAP-1);}
}
static void split(const char *line,ssize_t n,const char *a[51],size_t length[51]) {
    int col=0;a[0]=line;
    for(const char *p=line;p<line+n;p++)if(*p==','){
        if(col>=50)fail("too many columns");
        length[col]=(size_t)(p-a[col]);a[++col]=p+1;
    }
    if(col!=50)fail("column count changed");
    const char *end=line+n;
    while(end>a[50]&&(end[-1]=='\n'||end[-1]=='\r'))end--;
    length[50]=(size_t)(end-a[50]);
}
int main(int argc,char **argv){
    if(argc!=5)fail("usage: verify ACTIVE_PATCH.csv ORIGINAL.csv NEW.csv EXPECTED_SOURCE_SHA256");
    table=calloc(CAP,sizeof(Row));if(!table)fail("memory");
    FILE *patch=fopen(argv[1],"rb");if(!patch)fail("patch open");
    char *buf=NULL;size_t cap=0;ssize_t n=getline(&buf,&cap,patch);if(n<=0)fail("patch header");
    uint32_t keys=0;
    while((n=getline(&buf,&cap,patch))>0){
        char *part[6],*save=NULL;int count=0;
        for(char *p=strtok_r(buf,",\r\n",&save);p&&count<6;p=strtok_r(NULL,",\r\n",&save))part[count++]=p;
        if(count!=6)fail("patch structure");
        Row *r=get(idnum(part[0],part[0]+strlen(part[0])),daynum(part[1],part[1]+strlen(part[1])));
        if(r->id)fail("duplicate patch");
        r->id=idnum(part[0],part[0]+strlen(part[0]));r->day=daynum(part[1],part[1]+strlen(part[1]));
        for(int i=0;i<4;i++)if(!(r->values[i]=strdup(part[i+2])))fail("patch allocation");
        keys++;
    }
    fclose(patch);if(keys!=EXPECTED_KEYS)fail("key count");
    FILE *before=fopen(argv[2],"rb"),*after=fopen(argv[3],"rb");
    if(!before||!after)fail("CSV open");
    setvbuf(before,NULL,_IOFBF,16*1024*1024);setvbuf(after,NULL,_IOFBF,16*1024*1024);
    char *b=NULL,*a=NULL;size_t cb=0,ca=0;
    ssize_t nb=getline(&b,&cb,before),na=getline(&a,&ca,after);
    if(nb!=na||nb<=0||memcmp(b,a,(size_t)nb))fail("header changed");
    CC_SHA256_CTX source_hash,output_hash;
    CC_SHA256_Init(&source_hash);CC_SHA256_Init(&output_hash);
    CC_SHA256_Update(&source_hash,b,(CC_LONG)nb);
    CC_SHA256_Update(&output_hash,a,(CC_LONG)na);
    uint64_t rows=0,changed=0;
    while((nb=getline(&b,&cb,before))>0){
        na=getline(&a,&ca,after);if(na<=0)fail("output short");
        CC_SHA256_Update(&source_hash,b,(CC_LONG)nb);
        CC_SHA256_Update(&output_hash,a,(CC_LONG)na);
        const char *comma=memchr(b,',',(size_t)nb),*next=comma?memchr(comma+1,',',(size_t)(b+nb-comma-1)):NULL;
        if(!next)fail("bad original key");
        Row *r=get(idnum(comma+1,next),daynum(b,comma));
        if(!r->id){if(nb!=na||memcmp(b,a,(size_t)nb))fail("non-patch row changed");}
        else{
            const char *bs[51],*as[51];size_t bl[51],al[51];
            split(b,nb,bs,bl);split(a,na,as,al);
            for(int col=0;col<51;col++){
                int idx=col==8?0:col==10?1:col==50?2:col==38?3:-1;
                if(idx>=0){
                    if(strlen(r->values[idx])!=al[col]||memcmp(r->values[idx],as[col],al[col]))fail("replacement mismatch");
                }else if(bl[col]!=al[col]||memcmp(bs[col],as[col],bl[col]))fail("unrelated column changed");
            }
            r->hits++;changed++;
        }
        if(++rows%UINT64_C(20000000)==0)fprintf(stderr,"verified %" PRIu64 " rows\n",rows);
    }
    if(ferror(before)||getline(&a,&ca,after)>0||rows!=EXPECTED_ROWS)fail("row count/read");
    uint32_t seen=0;for(uint32_t i=0;i<CAP;i++)if(table[i].id&&table[i].hits)seen++;
    if(seen!=keys)fail("patch key not seen");
    unsigned char source_digest[32],output_digest[32];
    char source_hex[65],output_hex[65];const char *alphabet="0123456789abcdef";
    CC_SHA256_Final(source_digest,&source_hash);CC_SHA256_Final(output_digest,&output_hash);
    for(int i=0;i<32;i++){
        source_hex[2*i]=alphabet[source_digest[i]>>4];source_hex[2*i+1]=alphabet[source_digest[i]&15];
        output_hex[2*i]=alphabet[output_digest[i]>>4];output_hex[2*i+1]=alphabet[output_digest[i]&15];
    }
    source_hex[64]=0;output_hex[64]=0;
    if(strcmp(source_hex,argv[4]))fail("source SHA-256 mismatch");
    printf("PASS rows=%" PRIu64 " patched_rows=%" PRIu64 " matched_keys=%u source_sha256=%s output_sha256=%s\n",rows,changed,seen,source_hex,output_hex);
    free(a);free(b);free(buf);free(table);fclose(before);fclose(after);return 0;
}
