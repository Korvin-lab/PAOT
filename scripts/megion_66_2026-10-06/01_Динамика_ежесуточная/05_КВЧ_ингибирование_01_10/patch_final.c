#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CAP (1u<<18)
#ifndef ROWS
#define ROWS UINT64_C(138583872)
#endif
#ifndef KEYS
#define KEYS 71460
#endif
typedef struct {uint64_t id;uint32_t day,hits;char *kv,*ing;} Entry;
static Entry *table;
static void fail(const char *s){fprintf(stderr,"FAIL %s\n",s);exit(1);}
static uint64_t idnum(const char *a,const char *b){uint64_t x=0;if(a==b)fail("id");for(;a<b;a++){if(*a<'0'||*a>'9')fail("id digit");x=x*10+(*a-'0');}return x;}
static uint32_t daynum(const char *a,const char *b){if(b-a!=10||a[4]!='-'||a[7]!='-')fail("date");uint32_t x=0;for(int i=0;i<10;i++)if(i!=4&&i!=7){if(a[i]<'0'||a[i]>'9')fail("date digit");x=x*10+(a[i]-'0');}return x;}
static Entry *find(uint64_t id,uint32_t day){uint64_t h=id*UINT64_C(11400714819323198485)^(uint64_t)day*UINT64_C(14029467366897019727);uint32_t i=(uint32_t)(h^(h>>32))&(CAP-1);for(;;){Entry *e=&table[i];if(!e->id||(e->id==id&&e->day==day))return e;i=(i+1)&(CAP-1);}}
static int split(char *s,ssize_t n,char *a[51],char *b[51]){int c=0;a[0]=s;for(char *p=s;p<s+n;p++)if(*p==','){if(c>=50)fail("too many columns");b[c]=p;a[++c]=p+1;}if(c!=50)fail("column count");b[50]=s+n;return 51;}
static char *number(const char *a,const char *b,int positive){if(a==b)return NULL;char *s=strndup(a,(size_t)(b-a));if(!s)fail("alloc");char *e;double x=strtod(s,&e);if(*e||!isfinite(x)||(positive?x<=0:x<0))fail("number");return s;}
static void upd(CC_SHA256_CTX *h,const char *p,size_t n){if(n&&!CC_SHA256_Update(h,p,(CC_LONG)n))fail("sha");}
static void hex(CC_SHA256_CTX *h,char s[65]){unsigned char a[32];const char *d="0123456789abcdef";CC_SHA256_Final(a,h);for(int i=0;i<32;i++){s[2*i]=d[a[i]>>4];s[2*i+1]=d[a[i]&15];}s[64]=0;}
int main(int argc,char **argv){
 if(argc!=5)fail("usage PATCH SOURCE OUTPUT EXPECTED_SOURCE_SHA");table=calloc(CAP,sizeof(Entry));if(!table)fail("alloc table");
 FILE *patch=fopen(argv[1],"rb");if(!patch)fail("patch open");char *line=NULL;size_t cap=0;ssize_t n=getline(&line,&cap,patch);if(n<=0||strcmp(line,"id,date,kvch,ing_factor\n"))fail("patch header");
 uint32_t keys=0;while((n=getline(&line,&cap,patch))>0){char *b=strchr(line,','),*c=b?strchr(b+1,','):NULL,*d=c?strchr(c+1,','):NULL;if(!d)fail("patch columns");char *z=line+n;while(z>d&&(z[-1]=='\n'||z[-1]=='\r'))z--;if(memchr(d+1,',',(size_t)(z-d-1)))fail("patch extra");
  uint64_t id=idnum(line,b);uint32_t day=daynum(b+1,c);Entry *e=find(id,day);if(e->id)fail("duplicate patch");e->id=id;e->day=day;e->kv=number(c+1,d,1);e->ing=number(d+1,z,0);keys++;}
 if(ferror(patch)||fclose(patch)||keys!=KEYS)fail("patch count");
 FILE *src=fopen(argv[2],"rb"),*dst=fopen(argv[3],"wb");if(!src||!dst)fail("csv open");setvbuf(src,NULL,_IOFBF,16*1024*1024);setvbuf(dst,NULL,_IOFBF,16*1024*1024);
 CC_SHA256_CTX ih,oh;CC_SHA256_Init(&ih);CC_SHA256_Init(&oh);char *out=NULL;size_t outcap=0;uint64_t rows=0,kvr=0,ingr=0;
 while((n=getline(&line,&cap,src))>0){upd(&ih,line,(size_t)n);if(memchr(line,'"',(size_t)n))fail("quoted csv");char *a[51],*b[51];split(line,n,a,b);
  Entry *e=NULL;if(rows){e=find(idnum(a[1],b[1]),daynum(a[0],b[0]));if(!e->id)fail("unmapped key");if(a[7]!=b[7]||a[11]!=b[11]||a[44]!=b[44])fail("original target not blank");e->hits++;kvr+=e->kv!=NULL;ingr+=e->ing!=NULL;}
  else {if(strncmp(line,"\xef\xbb\xbf" "date,id,",11)||strncmp(a[7],"kvch,",5)||strncmp(a[11],"ing_factor,",11)||strncmp(a[44],"seg_v_mix_source_techregime,",28))fail("header");}
  const char *v7=rows?(e->kv?e->kv:""):"kvch",*v11=rows?(e->ing?e->ing:""):"ing_factor";
  size_t x=(size_t)(a[7]-line),y=(size_t)(a[11]-b[7]),z=(size_t)(a[44]-b[11]),w=(size_t)(line+n-(b[44]+1));size_t need=x+strlen(v7)+y+strlen(v11)+z+w;
  if(need>outcap){outcap=need+4096;out=realloc(out,outcap);if(!out)fail("out alloc");}char *q=out;
  memcpy(q,line,x);q+=x;memcpy(q,v7,strlen(v7));q+=strlen(v7);memcpy(q,b[7],y);q+=y;memcpy(q,v11,strlen(v11));q+=strlen(v11);memcpy(q,b[11],z);q+=z;memcpy(q,b[44]+1,w);q+=w;
  if((size_t)(q-out)!=need||fwrite(out,1,need,dst)!=need)fail("write");upd(&oh,out,need);if(rows&&rows%UINT64_C(10000000)==0)fprintf(stderr,"patched %" PRIu64 " rows\n",rows);rows++;
 }
 if(ferror(src)||rows!=ROWS+1)fail("row count");uint32_t seen=0;for(uint32_t i=0;i<CAP;i++)if(table[i].id&&table[i].hits)seen++;if(seen!=KEYS)fail("missing keys");
 if(fflush(dst)||fsync(fileno(dst))||fclose(dst)||fclose(src))fail("close");char hs[65],ho[65];hex(&ih,hs);hex(&oh,ho);if(strcmp(hs,argv[4]))fail("source SHA mismatch");
 printf("PASS rows=%" PRIu64 " keys=%u kvch_segment_rows=%" PRIu64 " ing_segment_rows=%" PRIu64 " source_sha256=%s output_sha256=%s\n",rows-1,seen,kvr,ingr,hs,ho);free(out);free(line);free(table);return 0;
}
