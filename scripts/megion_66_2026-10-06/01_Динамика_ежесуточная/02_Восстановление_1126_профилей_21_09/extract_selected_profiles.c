#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_string(const void *a, const void *b) {
    const char *const *left = a, *const *right = b;
    return strcmp(*left, *right);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr,"usage: %s INPUT.csv KEYS.tsv OUTPUT.csv\n",argv[0]); return 2; }
    FILE *keys=fopen(argv[2],"rb"), *in=fopen(argv[1],"rb"), *out=fopen(argv[3],"wb");
    if(!keys||!in||!out){perror("open");return 2;}
    char **need=NULL,*line=NULL; size_t cap=0,n_need=0,need_cap=0; ssize_t n;
    while((n=getline(&line,&cap,keys))>=0){
        char *tab=strchr(line,'\t'); if(!tab) continue;
        *tab='\0'; char *day=tab+1; day[strcspn(day,"\r\n")]='\0';
        char *key=malloc(strlen(line)+strlen(day)+2); sprintf(key,"%s,%s",day,line);
        if(n_need==need_cap){need_cap=need_cap?need_cap*2:256;need=realloc(need,need_cap*sizeof(*need));}
        need[n_need++]=key;
    }
    qsort(need,n_need,sizeof(*need),cmp_string);
    rewind(in); free(line);line=NULL;cap=0;
    if((n=getline(&line,&cap,in))<0){fprintf(stderr,"empty input\n");return 1;} fputs(line,out);
    unsigned long long rows=0,selected_rows=0,selected_profiles=0; char previous[128]={0};
    while((n=getline(&line,&cap,in))>=0){
        rows++; char *first=strchr(line,','),*second=first?strchr(first+1,','):NULL;
        if(!second){fprintf(stderr,"bad row %llu\n",rows+1);return 1;}
        size_t length=(size_t)(second-line); if(length>=sizeof(previous)){fprintf(stderr,"oversize key\n");return 1;}
        char current[128]; memcpy(current,line,length);current[length]='\0';
        char *probe=current; char **found=bsearch(&probe,need,n_need,sizeof(*need),cmp_string);
        if(found){
            fputs(line,out); selected_rows++;
            if(strcmp(current,previous)!=0){strcpy(previous,current);selected_profiles++;}
        }
    }
    fprintf(stdout,"source_rows=%llu selected_profiles=%llu selected_rows=%llu\n",rows,selected_profiles,selected_rows);
    for(size_t i=0;i<n_need;i++)free(need[i]);free(need);free(line);fclose(keys);fclose(in);fclose(out);return 0;
}
