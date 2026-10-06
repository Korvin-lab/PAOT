#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLS 51
typedef struct { long key; int used; double sum[COLS]; unsigned int count[COLS]; } Segment;
typedef struct { char id[32]; Segment *table; size_t cap; } Pipe;

static unsigned long hash_long(unsigned long x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; return x; }
static Segment *segment_for(Pipe *pipe, long key) {
    size_t pos=hash_long((unsigned long)key)%pipe->cap;
    for (;;) { Segment *s=&pipe->table[pos]; if (!s->used) {s->used=1;s->key=key;return s;} if (s->key==key)return s; pos=(pos+1)%pipe->cap; }
}
static int cmp_segment(const void *a,const void*b){const Segment *x=*(Segment**)a,*y=*(Segment**)b;return (x->key>y->key)-(x->key<y->key);}

int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: %s INPUT.csv EDGE_PIPES.txt OUTPUT.csv\n",argv[0]);return 2;}
    FILE *list=fopen(argv[2],"rb"),*in=fopen(argv[1],"rb"),*out=fopen(argv[3],"wb");if(!list||!in||!out){perror("open");return 2;}
    Pipe pipes[16]={0};int pipes_n=0;char *line=NULL;size_t cap=0;ssize_t n;
    while((n=getline(&line,&cap,list))>=0){line[strcspn(line,"\r\n")]=0;if(!*line)continue;strncpy(pipes[pipes_n].id,line,31);pipes[pipes_n].cap=16384;pipes[pipes_n].table=calloc(pipes[pipes_n].cap,sizeof(Segment));if(!pipes[pipes_n].table){perror("calloc");return 2;}pipes_n++;}
    if((n=getline(&line,&cap,in))<0)return 1;fputs(line,out);
    unsigned long long rows=0,selected=0;char *fields[COLS];
    while((n=getline(&line,&cap,in))>=0){
        rows++;int field_n=0;fields[field_n++]=line;for(char *c=line;*c;c++)if(*c==','){*c=0;if(field_n<COLS)fields[field_n++]=c+1;}if(field_n!=COLS){fprintf(stderr,"bad fields at %llu\n",rows+1);return 1;}
        int index=-1;for(int p=0;p<pipes_n;p++)if(!strcmp(fields[1],pipes[p].id)){index=p;break;}if(index<0)continue;
        char *end=NULL;double distance=strtod(fields[18],&end);if(end==fields[18]){fprintf(stderr,"bad segment\n");return 1;}Segment *s=segment_for(&pipes[index],lround(distance*1000.0));selected++;
        for(int c=2;c<COLS;c++){if(c==18||fields[c][0]==0)continue;double value=strtod(fields[c],&end);if(end!=fields[c]&&isfinite(value)){s->sum[c]+=value;s->count[c]++;}}
    }
    for(int p=0;p<pipes_n;p++){
        size_t total=0;for(size_t i=0;i<pipes[p].cap;i++)if(pipes[p].table[i].used)total++;Segment **items=malloc(total*sizeof(*items));size_t at=0;for(size_t i=0;i<pipes[p].cap;i++)if(pipes[p].table[i].used)items[at++]=&pipes[p].table[i];qsort(items,total,sizeof(*items),cmp_segment);
        for(size_t i=0;i<total;i++){Segment*s=items[i];fprintf(out,"MEAN,%s",pipes[p].id);for(int c=2;c<COLS;c++){if(c==18){fprintf(out,",%.17g",s->key/1000.0);}else if(s->count[c])fprintf(out,",%.17g",s->sum[c]/s->count[c]);else fputc(',',out);}fputc('\n',out);}free(items);free(pipes[p].table);
    }
    fprintf(stdout,"source_rows=%llu selected_rows=%llu pipes=%d\n",rows,selected,pipes_n);free(line);fclose(list);fclose(in);fclose(out);return 0;
}
