#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {unsigned char R,G,B;
               } RGBstruct;

RGBstruct POHcols[30];

int readcolors(char *colormapfile);

int main(int argc, char *argv[])
{
  
  RGBstruct maskcols[3],watercol,outcol,col;

  char *masknames[3],*mapdir,*colname,*colmapname,
    hdrbuf[200];

  unsigned char *masks[3],
                *outarr,
                *poharr,
                POHval,maskval;

  long int masksize=4273500,i,l,x,y,c,xdim=1628,ydim=2625,m,POHc,N,NN;

  FILE *POHFILE, *OUTFILE, *MASKS[3],*COLFILE;

  setbuf(stdout,NULL);

  outarr=malloc(masksize*3);
  poharr=malloc(masksize);


  masknames[0]=getenv("WATERMASK");  
  masknames[1]=getenv("ROADMASK");  
  masknames[2]=getenv("NAMEMASK");
  colname=getenv("COLORFILE");
  colmapname=getenv("COLORMAPFILE");

  readcolors(colmapname);
  COLFILE=fopen(colname,"r");
  
  c=0;
  while(1)
  {
    fscanf(COLFILE,"%hu %hu %hu",&col.R,&col.G,&col.B);
    if(feof(COLFILE)) break;
    if(c==0) watercol=col;
    if(c>0) maskcols[c]=col;
    c++;
  }
  

  POHFILE=fopen(argv[1],"r");
  for(l=0;l<3;l++)
  {
       fgets(hdrbuf,199,POHFILE);
       if(hdrbuf[0]=='#') l--;
  }
  fread(poharr,masksize,1,POHFILE);
  fclose(POHFILE);

  OUTFILE=fopen(argv[2],"w");
  fprintf(OUTFILE,"P6\n1628 2625\n255\n");


  for(i=0;i<3;i++)  
  {
     MASKS[i]=fopen(masknames[i],"r");
     masks[i]=malloc(masksize);
     for(l=0;l<3;l++)
     {
       fgets(hdrbuf,199,MASKS[i]);
       if(hdrbuf[0]=='#') l--;
     }
     fread(&masks[i][0],masksize,1,MASKS[i]);
     fclose(MASKS[i]);
  }


  for(y=0;y<ydim;y++)
  {
     for(x=0;x<xdim;x++)
     {
       N=y*xdim+x;
       NN=3*N; 
       POHval=poharr[N];
       if(!POHval || POHval==255)
       {
          for(m=0;m<3;m++)
          {
	    {
               maskval=masks[m][N];
	       if(m==0 && !maskval) outcol=watercol; 
               if(maskval) outcol=maskcols[m+1];
            }
          }
       }
       else
       { 
         POHc=POHval/5;
         if(POHc>19) POHc=19; 
         outcol=POHcols[POHc];
       }
         if(POHval==255) 
	 {
	   outcol.R-=outcol.R/4;	   
           outcol.G-=outcol.G/4;
           outcol.B-=outcol.B/4;
	 }

       outarr[NN]=outcol.R;
       outarr[NN+1]=outcol.G;
       outarr[NN+2]=outcol.B;
     }
  }

  fwrite(outarr,masksize*3,1,OUTFILE);
  fclose(OUTFILE);

  return(0);
} 

int readcolors(char *colormapfile)
{
  unsigned char col[3],ocol[3],*linebuf;
  char buf[200];
  int i,xdim,ydim,nextcol,x,n,c;
  FILE *PPMF;

  c=0;

  PPMF=fopen(colormapfile,"r");
  for(i=0;i<3;i++)
  {  
    fgets(buf,199,PPMF);
    if(buf[0]=='#') i--;
    if(i==1) sscanf(buf,"%d %d",&xdim,&ydim);
  }
  linebuf=malloc(xdim*3);
  fread(linebuf,xdim*3,1,PPMF);
  n=0;
  for(x=0;x<xdim;x++)
  {
    n=3*x;
    for(i=0;i<3;i++) col[i]=linebuf[n+i];
    nextcol=0;
    if(x)
    {
       for(i=0;i<3;i++) 
       {
	 if(col[i]!=ocol[i]) 
         {
           nextcol=1;
           break;
	 }
       }
    } else nextcol=1;

    if(nextcol)
    { 
      POHcols[c].R=col[0];
      POHcols[c].G=col[1];
      POHcols[c].B=col[2];
      /*      printf("%2d %03d %03d %03d\n",c,col[0],col[1],col[2]); */
      c++;
    }
      
    for(i=0;i<3;i++) ocol[i]=col[i];
  }
  fclose(PPMF);
  free(linebuf);

  return(0);
}
