#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){if(argc!=3)return 2;FILE*a=fopen(argv[1],"rb"),*b=fopen(argv[2],"rb");if(!a||!b)return 2;unsigned char x[1024*1024],y[1024*1024];unsigned long long total=0;for(;;){size_t n=fread(x,1,sizeof(x),a);if(!n)break;size_t m=fread(y,1,n,b);if(m!=n||memcmp(x,y,n)){fprintf(stderr,"DIFF at byte %llu\n",total);return 1;}total+=n;}printf("PASS bytes=%llu\n",total);return 0;}
