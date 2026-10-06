#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CAP (1u<<18)
#ifndef ROWS
#define ROWS UINT64_C(138583872)
#endif
#ifndef KEYS
#define KEYS 71460
#endif
typedef struct {uint64_t id;uint32_t day,hits;char *value;} Entry;
static Entry *table;
static void fail(const char *s){fprintf(stderr,"VERIFY FAIL: %s\n",s);exit(1);}
static uint64_t idnum(const char *a,const char *b){uint64_t x=0;if(a==b)fail("id");for(;a<b;a++){if(*a<'0'||*a>'9')fail("id");x=x*10+(*a-'0');}return x;}
static uint32_t daynum(const char *a,const char *b){if(b-a!=10||a[4]!='-'||a[7]!='-')fail("date");uint32_t x=0;for(int i=0;i<10;i++)if(i!=4&&i!=7){if(a[i]<'0'||a[i]>'9')fail("date");x=x*10+(a[i]-'0');}return x;}
static Entry *find(uint64_t id,uint32_t day){uint64_t h=id*UINT64_C(11400714819323198485)^(uint64_t)day*UINT64_C(14029467366897019727);uint32_t i=(uint32_t)(h^(h>>32))&(CAP-1);for(;;){Entry *e=&table[i];if(!e->id||(e->id==id&&e->day==day))return e;i=(i+1)&(CAP-1);}}
static void split(const char *line,ssize_t n,const char *start[51],size_t len[51]){
 int c=0;start[0]=line;
 for(const char *p=line;p<line+n;p++)if(*p==','){if(c>=50)fail("many columns");len[c]=(size_t)(p-start[c]);start[++c]=p+1;}
 if(c!=50)fail("column count");const char *end=line+n;while(end>start[50]&&(end[-1]=='\n'||end[-1]=='\r'))end--;len[50]=(size_t)(end-start[50]);
}
static void hexhash(CC_SHA256_CTX *h,char out[65]){unsigned char d[32];const char *a="0123456789abcdef";CC_SHA256_Final(d,h);for(int i=0;i<32;i++){out[2*i]=a[d[i]>>4];out[2*i+1]=a[d[i]&15];}out[64]=0;}
int main(int argc,char **argv){
 if(argc!=5)fail("usage PATCH.csv ORIGINAL.csv TEMP.csv EXPECTED_SOURCE_SHA256");
 table=calloc(CAP,sizeof(Entry));if(!table)fail("memory");FILE *p=fopen(argv[1],"rb");if(!p)fail("patch open");
 char *buf=NULL;size_t cb=0;ssize_t n=getline(&buf,&cb,p);if(n<=0||strcmp(buf,"id,date,Min\n"))fail("patch header");
 uint32_t keys=0;while((n=getline(&buf,&cb,p))>0){char *v[3],*save=NULL;int m=0;for(char *s=strtok_r(buf,",\r\n",&save);s&&m<3;s=strtok_r(NULL,",\r\n",&save))v[m++]=s;if(m!=3)fail("patch columns");Entry *e=find(idnum(v[0],v[0]+strlen(v[0])),daynum(v[1],v[1]+strlen(v[1])));if(e->id)fail("duplicate key");e->id=idnum(v[0],v[0]+strlen(v[0]));e->day=daynum(v[1],v[1]+strlen(v[1]));e->value=strdup(v[2]);if(!e->value)fail("memory");keys++;}fclose(p);if(keys!=KEYS)fail("keys count");
 FILE *before=fopen(argv[2],"rb"),*after=fopen(argv[3],"rb");if(!before||!after)fail("CSV open");setvbuf(before,NULL,_IOFBF,16*1024*1024);setvbuf(after,NULL,_IOFBF,16*1024*1024);
 char *b=NULL,*a=NULL;size_t bb=0,aa=0;ssize_t nb=getline(&b,&bb,before),na=getline(&a,&aa,after);
 if(nb<=0||nb!=na||memcmp(b,a,(size_t)nb))fail("header changed");
 CC_SHA256_CTX sh,oh;CC_SHA256_Init(&sh);CC_SHA256_Init(&oh);CC_SHA256_Update(&sh,b,(CC_LONG)nb);CC_SHA256_Update(&oh,a,(CC_LONG)na);
 uint64_t rows=0,changed=0;
 while((nb=getline(&b,&bb,before))>0){
  na=getline(&a,&aa,after);if(na<=0)fail("output too short");CC_SHA256_Update(&sh,b,(CC_LONG)nb);CC_SHA256_Update(&oh,a,(CC_LONG)na);
  const char *c=memchr(b,',',(size_t)nb),*d=c?memchr(c+1,',',(size_t)(b+nb-c-1)):NULL;if(!d)fail("key");Entry *e=find(idnum(c+1,d),daynum(b,c));
  if(!e->id){if(nb!=na||memcmp(b,a,(size_t)nb))fail("non-patched row changed");}
  else {
   const char *bs[51],*as[51];size_t bl[51],al[51];split(b,nb,bs,bl);split(a,na,as,al);
   for(int col=0;col<51;col++){
    if(col==9){if(al[col]!=strlen(e->value)||memcmp(as[col],e->value,al[col]))fail("Min mismatch");}
    else if(bl[col]!=al[col]||memcmp(bs[col],as[col],bl[col]))fail("unrelated column changed");
   }
   e->hits++;changed++;
  }
  if(++rows%UINT64_C(20000000)==0)fprintf(stderr,"verified %" PRIu64 " rows\n",rows);
 }
 if(ferror(before)||getline(&a,&aa,after)>0||rows!=ROWS)fail("row count");
 uint32_t seen=0;for(uint32_t i=0;i<CAP;i++)if(table[i].id&&table[i].hits)seen++;if(seen!=keys)fail("missing keys");
 char source[65],output[65];hexhash(&sh,source);hexhash(&oh,output);if(strcmp(source,argv[4]))fail("source SHA");
 printf("PASS rows=%" PRIu64 " changed_segment_rows=%" PRIu64 " keys=%u source_sha256=%s output_sha256=%s\n",rows,changed,seen,source,output);
 free(a);free(b);free(buf);free(table);fclose(before);fclose(after);return 0;
}
