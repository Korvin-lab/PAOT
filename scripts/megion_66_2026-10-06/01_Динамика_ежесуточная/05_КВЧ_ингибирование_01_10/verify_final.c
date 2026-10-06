#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef ROWS
#define ROWS UINT64_C(138583872)
#endif
#ifndef KEYS
#define KEYS 71460
#endif
#define CAP (1u<<18)
typedef struct {uint64_t id;uint32_t day,hits;char *kv,*ing;} Entry;
static Entry *tab;
static void fail(const char *s){fprintf(stderr,"VERIFY FAIL %s\n",s);exit(1);}
static uint64_t idn(const char *a,const char *b){uint64_t x=0;if(a==b)fail("id");for(;a<b;a++){if(*a<'0'||*a>'9')fail("id digit");x=x*10+(*a-'0');}return x;}
static uint32_t dn(const char *a,const char *b){if(b-a!=10||a[4]!='-'||a[7]!='-')fail("date");uint32_t x=0;for(int i=0;i<10;i++)if(i!=4&&i!=7){if(a[i]<'0'||a[i]>'9')fail("date digit");x=x*10+(a[i]-'0');}return x;}
static Entry *lookup(uint64_t id,uint32_t day){uint64_t h=id*UINT64_C(11400714819323198485)^(uint64_t)day*UINT64_C(14029467366897019727);uint32_t k=(uint32_t)(h^(h>>32))&(CAP-1);for(;;){Entry *e=&tab[k];if(!e->id||(e->id==id&&e->day==day))return e;k=(k+1)&(CAP-1);}}
static int fields(const char *s,ssize_t n,const char *a[51],size_t z[51]){int c=0;a[0]=s;for(const char *p=s;p<s+n;p++)if(*p==','){if(c>=50)fail("too many columns");z[c]=(size_t)(p-a[c]);a[++c]=p+1;}const char *e=s+n;while(e>a[c]&&(e[-1]=='\n'||e[-1]=='\r'))e--;z[c]=(size_t)(e-a[c]);return c+1;}
static void upd(CC_SHA256_CTX *h,const char *p,size_t n){if(n&&!CC_SHA256_Update(h,p,(CC_LONG)n))fail("sha");}
static void hex(CC_SHA256_CTX *h,char x[65]){unsigned char a[32];const char *d="0123456789abcdef";CC_SHA256_Final(a,h);for(int i=0;i<32;i++){x[2*i]=d[a[i]>>4];x[2*i+1]=d[a[i]&15];}x[64]=0;}
int main(int argc,char **argv){
 if(argc!=5)fail("usage PATCH SOURCE RESULT EXPECTED_SOURCE_SHA");tab=calloc(CAP,sizeof(Entry));if(!tab)fail("alloc");
 FILE *p=fopen(argv[1],"rb");if(!p)fail("patch");char *l=NULL;size_t lc=0;ssize_t n=getline(&l,&lc,p);if(n<=0||strcmp(l,"id,date,kvch,ing_factor\n"))fail("patch header");unsigned keys=0;
 while((n=getline(&l,&lc,p))>0){const char *v[4];size_t z[4];int c=fields(l,n,v,z);if(c!=4)fail("patch fields");Entry *e=lookup(idn(v[0],v[0]+z[0]),dn(v[1],v[1]+z[1]));if(e->id)fail("duplicate");e->id=idn(v[0],v[0]+z[0]);e->day=dn(v[1],v[1]+z[1]);e->kv=strndup(v[2],z[2]);e->ing=strndup(v[3],z[3]);if(!e->kv||!e->ing)fail("alloc value");keys++;}
 if(ferror(p)||fclose(p)||keys!=KEYS)fail("patch count");FILE *b=fopen(argv[2],"rb"),*a=fopen(argv[3],"rb");if(!b||!a)fail("csv open");setvbuf(b,NULL,_IOFBF,16*1024*1024);setvbuf(a,NULL,_IOFBF,16*1024*1024);
 char *bl=NULL,*al=NULL;size_t bc=0,ac=0;ssize_t bn,an;CC_SHA256_CTX bh,ah;CC_SHA256_Init(&bh);CC_SHA256_Init(&ah);uint64_t rows=0,kvr=0,ingr=0;
 while((bn=getline(&bl,&bc,b))>0){an=getline(&al,&ac,a);if(an<=0)fail("short result");upd(&bh,bl,(size_t)bn);upd(&ah,al,(size_t)an);
  const char *bs[51],*as[51];size_t bz[51],az[51];if(fields(bl,bn,bs,bz)!=51||fields(al,an,as,az)!=50)fail("column count");
  Entry *e=NULL;if(rows){e=lookup(idn(bs[1],bs[1]+bz[1]),dn(bs[0],bs[0]+bz[0]));if(!e->id)fail("unknown key");e->hits++;}
  for(int i=0;i<51;i++){if(i==44)continue;int j=i>44?i-1:i;
   if(rows&&i==7){if(az[j]!=strlen(e->kv)||memcmp(as[j],e->kv,az[j]))fail("kvch mismatch");if(az[j])kvr++;}
   else if(rows&&i==11){if(az[j]!=strlen(e->ing)||memcmp(as[j],e->ing,az[j]))fail("ing mismatch");if(az[j])ingr++;}
   else if(bz[i]!=az[j]||memcmp(bs[i],as[j],bz[i]))fail("unrelated field changed");
  }
  if(rows&&rows%UINT64_C(10000000)==0)fprintf(stderr,"verified %" PRIu64 " rows\n",rows);rows++;
 }
 if(ferror(b)||getline(&al,&ac,a)>0||rows!=ROWS+1)fail("row count");unsigned seen=0;for(unsigned i=0;i<CAP;i++)if(tab[i].id&&tab[i].hits)seen++;if(seen!=KEYS)fail("unseen keys");
 char hs[65],ho[65];hex(&bh,hs);hex(&ah,ho);if(strcmp(hs,argv[4]))fail("source sha");printf("PASS rows=%" PRIu64 " keys=%u kvch_segment_rows=%" PRIu64 " ing_segment_rows=%" PRIu64 " source_sha256=%s output_sha256=%s\n",rows-1,seen,kvr,ingr,hs,ho);
 free(l);free(bl);free(al);free(tab);fclose(b);fclose(a);return 0;
}
