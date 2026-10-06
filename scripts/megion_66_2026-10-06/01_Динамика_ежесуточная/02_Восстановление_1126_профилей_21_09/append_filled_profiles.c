#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long copy_file(FILE *out, const char *path, int count_lines) {
    FILE *in=fopen(path,"rb"); if(!in){perror(path);exit(2);} char buffer[1024*1024];size_t n;unsigned long long lines=0;
    while((n=fread(buffer,1,sizeof(buffer),in))>0){if(count_lines)for(size_t i=0;i<n;i++)if(buffer[i]=='\n')lines++;if(fwrite(buffer,1,n,out)!=n){perror("write");exit(2);}}
    fclose(in);return lines;
}
int main(int argc,char**argv){
 if(argc!=4){fprintf(stderr,"usage: %s INPUT.csv FILL_INDEX.tsv OUTPUT.csv\n",argv[0]);return 2;}
 FILE*idx=fopen(argv[2],"rb"),*out=fopen(argv[3],"wb");if(!idx||!out){perror("open");return 2;}
 unsigned long long source_rows=copy_file(out,argv[1],1); char*line=NULL;size_t cap=0;ssize_t n;unsigned long long added_rows=0,profiles=0;
 while((n=getline(&line,&cap,idx))>=0){char *p=strchr(line,'\t');if(!p)continue;p=strchr(p+1,'\t');if(!p)continue;p++;p[strcspn(p,"\r\n")]=0;added_rows+=copy_file(out,p,1);profiles++;}
 free(line);fclose(idx);fclose(out);printf("source_rows=%llu appended_profiles=%llu appended_rows=%llu output_rows=%llu\n",source_rows-1,profiles,added_rows,source_rows-1+added_rows);return 0;
}
