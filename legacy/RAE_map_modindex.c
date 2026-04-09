#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {unsigned char R,G,B;
               } RGBstruct;

RGBstruct HHIcols[30];

int readcolors(char *colormapfile);

int main(int argc, char *argv[])
{
  
  RGBstruct maskcols[3],watercol,outcol,col,ppmcol,black;

  char *masknames[3],*mapdir,*colname,*colmapname,
    hdrbuf[200];

  unsigned char *masks[3],
                *outarr,
                *ppmarr,
                *poharr,
                *zeroarr,
                HHIval,maskval,zeroval;

  long int masksize=4273500,i,l,x,y,c,xdim=1628,ydim=2625,m,HHIc,N,NN;
  int POH_max,colI_max;
  double col_factor;

  FILE *POHFILE, *OUTFILE, *PPMFILE, *MASKS[3],*COLFILE,*ZEROFILE;

  setbuf(stdout,NULL);

  black.R=0;
  black.G=0;
  black.B=0;


  outarr=malloc(masksize*3);
  ppmarr=malloc(masksize*3);
  poharr=malloc(masksize);
  zeroarr=malloc(masksize);


  masknames[0]=getenv("WATERMASK");  
  masknames[1]=getenv("ROADMASK");  
  masknames[2]=getenv("NAMEMASK");
  POH_max=atof(getenv("POH_MAX"))*100+1;
  colname=getenv("COLORFILE");
  colmapname=getenv("COLORMAPFILE");

  colI_max=readcolors(colmapname);
  col_factor=(double)POH_max/((double)colI_max+1)*0.999;

  COLFILE=fopen(colname,"r");
  
  c=0;
  while(1)
  {
    fscanf(COLFILE,"%hhu %hhu %hhu",&col.R,&col.G,&col.B);
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

  PPMFILE=fopen(argv[3],"w");
  fprintf(PPMFILE,"P6\n1628 2625\n255\n");

  ZEROFILE=fopen(argv[4],"r");
  for(l=0;l<3;l++)
  {
       fgets(hdrbuf,199,ZEROFILE);
       if(hdrbuf[0]=='#') l--;
  }
  fread(zeroarr,masksize,1,ZEROFILE);
  fclose(ZEROFILE);

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
       HHIval=poharr[N];
       zeroval=zeroarr[N];
       if(!HHIval || HHIval>=254)
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
         HHIc=(int)((double)HHIval/col_factor);
         while(1) /* Modified index by zero isotherm height [100 m] (Jari Tuovinen) */
	 {
	   if(zeroval<=12) { HHIc+=2; break; }
	   if(zeroval<=17) { HHIc++; break; }
	   if(zeroval<=35) break;
	   HHIc--; 
           if(HHIc<0) HHIc=0;
           break;
	 }
         if(HHIc>colI_max) HHIc=colI_max+1; 
         outcol=HHIcols[HHIc];
         ppmcol=HHIcols[HHIc];
       }
         if(HHIval==255) 
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
  free(poharr);
  free(zeroarr);
  for(i=0;i<3;i++) free(masks[i]);

  return(0);
} 

int readcolors(char *colormapfile)
{
  unsigned char col[3],ocol[3],*linebuf;
  char buf[200]={0};
  int i,xdim,ydim,nextcol,x,n,colI;
  FILE *PPMF;

  colI=0;

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
      HHIcols[colI].R=col[0];
      HHIcols[colI].G=col[1];
      HHIcols[colI].B=col[2];
      /*      printf("%2d %03d %03d %03d\n",c,col[0],col[1],col[2]); */
      colI++;
    }
      
    for(i=0;i<3;i++) ocol[i]=col[i];
  }
  fclose(PPMF);
  free(linebuf);

  return(colI-2);
}
