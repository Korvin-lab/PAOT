#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *in = fopen(argv[1], "rb"); if (!in) { perror("open"); return 2; }
    const size_t bs=8*1024*1024; unsigned char *b=malloc(bs); if(!b) return 2;
    unsigned long long row=0, bytes=0, current=0, max=0, max_row=0, long_rows=0;
    int commas=0, reported=0; size_t prefix_len=0; char prefix[501]={0}; size_t n;
    while((n=fread(b,1,bs,in))>0) for(size_t i=0;i<n;i++) {
        unsigned char c=b[i]; bytes++; current++;
        if(prefix_len<500) prefix[prefix_len++]=(char)c;
        if(c==',') commas++;
        if(c=='\n') {
            row++;
            if(current>max){max=current;max_row=row;}
            if(current>5000){
                long_rows++;
                if(!reported){prefix[prefix_len]=0; printf("FIRST_LONG row=%llu bytes=%llu commas=%d prefix=%s\n",row,current,commas,prefix);reported=1;}
            }
            current=0;commas=0;prefix_len=0;
        }
    }
    printf("rows=%llu bytes=%llu max_line_bytes=%llu max_row=%llu long_rows=%llu trailing=%llu\n",row,bytes,max,max_row,long_rows,current);
    free(b);fclose(in);return 0;
}
