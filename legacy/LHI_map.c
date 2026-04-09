#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {unsigned char R,G,B;
               } RGBstruct;

RGBstruct LHIcols[30];

int readcolors(char *colormapfile);

int main(int argc, char *argv[])
{
  
  RGBstruct maskcols[3],watercol,outcol,col,ppmcol,black;

  char *masknames[3],*mapdir,*colname,*colmapname,
    hdrbuf[300];

  double LHIh,MinLHIh,MaxLHIh,LHIint,LHIrange;

  unsigned char *masks[3],
                *outarr,
                *lhiarr,
                *ppmarr,
                LHIval,maskval;

  long int masksize=4273500,max_colorind,i,l,x,y,c,xdim=1628,ydim=2625,m,LHIc,N,NN;

  FILE *LHIFILE, *OUTFILE, *MASKS[3],*COLFILE,*PPMFILE;

  setbuf(stdout,NULL);

  black.R=0;
  black.G=0;
  black.B=0;

  outarr=malloc(masksize*3);
  ppmarr=malloc(masksize*3);
  lhiarr=malloc(masksize);


  masknames[0]=getenv("WATERMASK");  
  masknames[1]=getenv("ROADMASK");  
  masknames[2]=getenv("NAMEMASK");
  colname=getenv("COLORFILE");
  colmapname=getenv("COLORMAPFILE");

  max_colorind=readcolors(colmapname);
  COLFILE=fopen(colname,"r");
  
  c=0;
  while(1)
  {
    fscanf(COLFILE,"%d %d %d",&col.R,&col.G,&col.B);
    if(feof(COLFILE)) break;
    if(c==0) watercol=col;
    if(c>0) maskcols[c]=col;
    c++;
  }
  fclose(COLFILE);
  

  LHIFILE=fopen(argv[1],"r");
  for(l=0;l<3;l++)
  {
       memset(hdrbuf,0,300);
       fgets(hdrbuf,299,LHIFILE);
       if(hdrbuf[0]=='#') l--;
  }
  fread(lhiarr,masksize,1,LHIFILE);
  fclose(LHIFILE);

  OUTFILE=fopen(argv[2],"w");
  fprintf(OUTFILE,"P6\n1628 2625\n255\n");

  PPMFILE=fopen(argv[3],"w");
  fprintf(PPMFILE,"P6\n1628 2625\n255\n");

  for(i=0;i<3;i++)  
  {
     MASKS[i]=fopen(masknames[i],"r");
     masks[i]=malloc(masksize);
     for(l=0;l<3;l++)
     {
       memset(hdrbuf,0,300);
       fgets(hdrbuf,299,MASKS[i]);
       if(hdrbuf[0]=='#') l--;
     }
     fread(&masks[i][0],masksize,1,MASKS[i]);
     fclose(MASKS[i]);
  }

  /*
  MaxLHIh=45.0;
  MinLHIh=-45.0;
  */
  MaxLHIh=55.0;
  MinLHIh=-20.0;
  LHIrange=MaxLHIh-MinLHIh;
  LHIint=((double)max_colorind + 1.0)/LHIrange;

  for(y=0;y<ydim;y++)
  {
     for(x=0;x<xdim;x++)
     {
       N=y*xdim+x;
       NN=3*N; 
       LHIval=lhiarr[N];
       LHIh=(double)((int)LHIval-100); /* 50 dBZ TOP - 20C isotermin erotus [100m] */
       if(!LHIval || LHIval>=254)
       {
	  ppmcol=black;
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
         if(LHIh<MinLHIh) LHIc=0;
         if(LHIh>=MaxLHIh) LHIc=max_colorind;
         if(LHIh>=MinLHIh && LHIh<MaxLHIh) LHIc=(int)((LHIh-MinLHIh)*LHIint);
         outcol=LHIcols[LHIc];
         ppmcol=LHIcols[LHIc];
       }

       if(LHIval==255) 
       {
	   outcol.R-=outcol.R/4;	   
           outcol.G-=outcol.G/4;
           outcol.B-=outcol.B/4;
       }

       outarr[NN]=outcol.R;
       outarr[NN+1]=outcol.G;
       outarr[NN+2]=outcol.B;      

       ppmarr[NN]=ppmcol.R;
       ppmarr[NN+1]=ppmcol.G;
       ppmarr[NN+2]=ppmcol.B;

     }
  }

  fwrite(outarr,masksize*3,1,OUTFILE);
  fwrite(ppmarr,masksize*3,1,PPMFILE);
  fclose(OUTFILE);
  fclose(PPMFILE);
  free(outarr);
  free(ppmarr);
  free(lhiarr);
  for(i=0;i<3;i++) free(masks[i]);
  return(0);
} 

int readcolors(char *colormapfile)
{
  unsigned char col[3],ocol[3],*linebuf;
  char buf[300];
  int i,xdim,ydim,nextcol,x,n,c;
  FILE *PPMF;

  c=0;

  PPMF=fopen(colormapfile,"r");
  for(i=0;i<3;i++)
  {  
    memset(buf,0,300);
    fgets(buf,299,PPMF);
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
      LHIcols[c].R=col[0];
      LHIcols[c].G=col[1];
      LHIcols[c].B=col[2];
      /*      printf("%2d %03d %03d %03d\n",c,col[0],col[1],col[2]); */
      c++;
    }
      
    for(i=0;i<3;i++) ocol[i]=col[i];
  }
  fclose(PPMF);
  free(linebuf);

  return(c-1);
}
