#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char pipe[32], day[16], path[1024]; } Fill;
static int cmp_fill(const void *a,const void*b){const Fill*x=a,*y=b;int c=strcmp(x->pipe,y->pipe);return c?c:strcmp(x->day,y->day);}
static int cmp_key(const char *pipe,const char *day,const Fill *f){int c=strcmp(pipe,f->pipe);return c?c:strcmp(day,f->day);}
static unsigned long long copy_profile(FILE *out,const char *path){FILE *in=fopen(path,"rb");if(!in){perror(path);exit(2);}char b[1024*1024];size_t n;unsigned long long lines=0;while((n=fread(b,1,sizeof(b),in))>0){for(size_t i=0;i<n;i++)if(b[i]=='\n')lines++;if(fwrite(b,1,n,out)!=n){perror("write");exit(2);}}fclose(in);return lines;}
int main(int argc,char**argv){
 if(argc!=4){fprintf(stderr,"usage: %s INPUT.csv FILL_INDEX.tsv OUTPUT.csv\n",argv[0]);return 2;}
 FILE*idx=fopen(argv[2],"rb"),*in=fopen(argv[1],"rb"),*out=fopen(argv[3],"wb");if(!idx||!in||!out){perror("open");return 2;}
 Fill *fills=NULL;size_t nf=0,cap=0;char*line=NULL;size_t lcap=0;ssize_t n;
 while((n=getline(&line,&lcap,idx))>=0){char*p=strtok(line,"\t\r\n"),*d=strtok(NULL,"\t\r\n"),*path=strtok(NULL,"\r\n");if(!p||!d||!path){fprintf(stderr,"bad index\n");return 1;}if(nf==cap){cap=cap?cap*2:2048;fills=realloc(fills,cap*sizeof(*fills));}snprintf(fills[nf].pipe,sizeof(fills[nf].pipe),"%s",p);snprintf(fills[nf].day,sizeof(fills[nf].day),"%s",d);snprintf(fills[nf].path,sizeof(fills[nf].path),"%s",path);nf++;}
 qsort(fills,nf,sizeof(*fills),cmp_fill);rewind(in);free(line);line=NULL;lcap=0;
 if((n=getline(&line,&lcap,in))<0){fprintf(stderr,"empty input\n");return 1;}fputs(line,out);
 size_t next=0;unsigned long long source_rows=0,source_profiles=0,inserted_rows=0,inserted_profiles=0;char prev_pipe[32]={0},prev_day[16]={0};
 while((n=getline(&line,&lcap,in))>=0){source_rows++;char *a=strchr(line,','),*b=a?strchr(a+1,','):NULL;if(!b){fprintf(stderr,"bad source row %llu\n",source_rows+1);return 1;}char day[16],pipe[32];size_t dl=(size_t)(a-line),pl=(size_t)(b-a-1);if(dl>=sizeof(day)||pl>=sizeof(pipe)){fprintf(stderr,"key too long\n");return 1;}memcpy(day,line,dl);day[dl]=0;memcpy(pipe,a+1,pl);pipe[pl]=0;
 if(strcmp(pipe,prev_pipe)||strcmp(day,prev_day)){
   if(source_profiles && (strcmp(pipe,prev_pipe)<0 || (!strcmp(pipe,prev_pipe)&&strcmp(day,prev_day)<0))){fprintf(stderr,"source profiles not sorted\n");return 1;}
   source_profiles++;strcpy(prev_pipe,pipe);strcpy(prev_day,day);
   while(next<nf && cmp_key(pipe,day,&fills[next])>0){inserted_rows+=copy_profile(out,fills[next].path);inserted_profiles++;next++;}
   if(next<nf && cmp_key(pipe,day,&fills[next])==0){fprintf(stderr,"fill collides with existing profile %s %s\n",pipe,day);return 1;}
 }
 fputs(line,out);
 }
 while(next<nf){inserted_rows+=copy_profile(out,fills[next].path);inserted_profiles++;next++;}
 fprintf(stdout,"source_rows=%llu source_profiles=%llu inserted_rows=%llu inserted_profiles=%llu output_rows=%llu\n",source_rows,source_profiles,inserted_rows,inserted_profiles,source_rows+inserted_rows);
 free(line);free(fills);fclose(idx);fclose(in);fclose(out);return 0;
}
