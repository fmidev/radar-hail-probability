# define _XOPEN_SOURCE
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>
# include <sys/stat.h>
# include <sys/types.h>
# include <stdint.h>
# include <unistd.h>

#define RADN 21

time_t sec_from_date(char *stim);
void date_from_sec(char *datestr,time_t secs);

int main(int argc, char *argv[])
{
  time_t T,Tb,Te,Ti,Th;
  int lon_0=25,lat_0=90,lat_ts=60,isotherm;
  long Xdim=0,Ydim=0,Tdim=0;
  char *data,*p,*q,hstr[20],datestr[20],*name,
    levelname[300]={0},radlevelname[200],pgmname[200];
  char linebuf[10000]={0};
  long int i,j,h,t,x,y,hlen,fields,hours,X,Y,
    gX=36,gY=35,NX,NY,nx,ny,ix,iy,R,R2,rx,ry,r2,
    outX=929,outY=1571,reqsize=0,datasize;
  double height,***heights,**intgrid,dh,rc,hs,rcs;
  double SWlon=17.1,SWlat=56.8,NElon=38.0,NElat=71.0;
  struct stat statbuf;
  unsigned char ch,**pgmarr;
  float fheight,sfheight;
  uint32_t iheight;
  long int size,rawsize,N;
  int Yb,Mb,Db,hb,mb,Ye,Me,De,he,me;
  int radxy[RADN][2],rI;
  char radars[RADN][4]={"KOR","VAN","ANJ","IKA","KES","PET","KUO","VIM","NUR","UTA","LUO","GOT","ARL","HUD","OVI","LUL","KIR","KER","VIH","KAN","KAU"};
  char dates[100][13]={{0}};
  FILE *LISTF,*PGMF,*HGTRADS,*RADHGT,*RAWPGM;
  int offp=0,reqf=0;

  setbuf(stdout,NULL);

  /* Koordinaatit FIN1000 -projektiosta +proj=stere +a=6371288 +lon_0=25E +lat_0=90N +lat_ts=60 +x_0=487139.27 +y_0=3510623.51 */
  radxy[0][0]=302; radxy[0][1]=345;
  radxy[1][0]=481; radxy[1][1]=355;
  radxy[2][0]=602; radxy[2][1]=428;
  radxy[3][0]=388; radxy[3][1]=523;
  radxy[4][0]=735; radxy[4][1]=550;
  radxy[5][0]=510; radxy[5][1]=580;
  radxy[6][0]=608; radxy[6][1]=644;
  radxy[7][0]=429; radxy[7][1]=668;
  radxy[8][0]=701; radxy[8][1]=756;
  radxy[9][0]=550; radxy[9][1]=852;
  radxy[10][0]=568; radxy[10][1]=1108;
  radxy[11][0]=85; radxy[11][1]=39;
  radxy[12][0]=92; radxy[12][1]=311;
  radxy[13][0]=53; radxy[13][1]=532;
  radxy[14][0]=168; radxy[14][1]=746;
  radxy[15][0]=358; radxy[15][1]=938;
  radxy[16][0]=309; radxy[16][1]=1177;
  radxy[17][0]=494; radxy[17][1]=368;
  radxy[18][0]=460; radxy[18][1]=387;
  radxy[19][0]=357; radxy[19][1]=528;
  radxy[20][0]=584; radxy[20][1]=1248;
  dh=255.0/3000.0;

  R=atol(argv[2]);
  name=argv[3];
  if(strstr(name,"ZERO")) isotherm=0;
  if(strstr(name,"M20")) isotherm=-20;

  { /* read txt file */

     FILE *TXTF;
     char *p,linebuf[10000]={0};
     char *level,*VERB=NULL;
     int t=0,ot=-1,x,y,items;
     double *datavec;
     float fval;
     long N=0;
     size_t size;
     struct stat statbuf;

     VERB=getenv("VERBOSE");
     stat(argv[1],&statbuf);
     size=statbuf.st_size;
     datavec=calloc(size,sizeof(double));

     TXTF=fopen(argv[1],"r");
     level=argv[2];
     fgets(linebuf,9999,TXTF);

     while(1)
     {
        memset(linebuf,0,10000);
        fgets(linebuf,9999,TXTF);
        if(feof(TXTF)) break;
        /* printf("%s",linebuf); */
        if(t!=ot)
        {
           sprintf(dates[t],"%.10s00",linebuf);
           if(VERB) printf("TIMESTAMP %s\n",dates[t]);
           fgets(linebuf,9999,TXTF);
           ot=t;
           y=0;
           continue;
        }

        p=linebuf;
        x=-1;
        while(1)
        {
           if(p[0]=='\n') 
           {
	       Ydim=y;
               y=0;
               t++; 
               break;
           }
           while(p[0]==' ') { p++; continue; }

	/*        items=sscanf(p,"%s",sval); */
	   items=sscanf(p,"%f",&fval);
           if(!items) fval=0.0;
           x++;
           datavec[N]=(double)fval;
	   if(VERB) printf("%d\t%d\t%d\t%.2f\n",t,y,x,datavec[N]);
           N++;
           p=strchr(p,' ');
           if(!p) 
           {
	       Xdim=x+1;
               y++; 
               break;
           }
        }
     }
     Tdim=t;

     /*     printf("%d %d %d",Tdim, Ydim,Xdim);  */

     heights=calloc(Tdim,sizeof(double **));
     for(t=0;t<Tdim;t++)
     {
         heights[t]=calloc(Ydim,sizeof(double *));
	 for(y=0;y<Ydim;y++) heights[t][y]=calloc(Xdim,sizeof(double));
	 for(y=0;y<Ydim;y++)
	 {
            for(x=0;x<Xdim;x++)
            {
               N=t*Ydim*Xdim + y*Xdim + x; 
               heights[t][Ydim-y-1][x]=datavec[N];
	    }
	 }
     }

     fclose(TXTF);
     free(datavec);
  }
     



  gX=36;
  gY=35;
  NX=gX*Xdim;
  NY=gY*Ydim;
  R2=R*R*gY*gX;

  printf("TOIMII\n");
  printf("%ld %ld\n",Xdim,Ydim);
  rawsize=Xdim*Ydim;

  /* varataan muistia interplointihilalle */

  intgrid=malloc(NX*sizeof(double *));
  for(nx=0;nx<NX;nx++)
  {
    intgrid[nx]=malloc(NY*sizeof(double));
  }
  pgmarr=malloc(NY*sizeof(unsigned char *));
  for(ny=0;ny<NY;ny++)  pgmarr[ny]=malloc(NX*sizeof(unsigned char));


  /* Käydään aikakentät läpi */

  for(Ti=0;Ti<Tdim;Ti++)
  {
    T=sec_from_date(dates[Ti]);
    sprintf(levelname,"%s_%s_levels.dat",dates[Ti],name);
    LISTF=fopen(levelname,"w");
    N=0;
    for(y=0;y<Ydim;y++)
    { 
       for(x=0;x<Xdim;x++) 
       {
          fprintf(LISTF,"%4ld %4ld %7.1f\n",x,y,heights[Ti][y][x]);
       }
       fprintf(LISTF,"\n");
    }
    fclose(LISTF);

    for(ny=0;ny<NY;ny++) 
    {
    y=ny/gY;
    for(nx=0;nx<NX;nx++)
    {
      x=nx/gX;
      hs=0.0;
      rcs=0.0;
      for(j=-R;j<=R;j++) 
      {
	 iy=y+j;
         if(iy<0 || iy>=Ydim) continue;
         ry=iy*gY-ny;
         for(i=-R;i<=R;i++)
         {
	    ix=x+i;
            if(ix<0 || ix>=Xdim) continue;
            rx=ix*gX-nx;
            r2=rx*rx+ry*ry;
            if(r2>R2) continue;
            rc=R2-r2;
            rcs+=rc;
            hs+=heights[Ti][iy][ix]*(double)rc;
         }
      }
      if(R) height=hs/(double)rcs;
      else height=heights[Ti][y][x];  
      intgrid[nx][ny]=height;
      ch=height/100+1;
      pgmarr[ny][nx]=ch;
    }
    }
   
    sprintf(pgmname,"%s_%s_level.pgm",dates[Ti],name); 
    PGMF=fopen(pgmname,"w");
    fprintf(PGMF,"P5\n");
    fprintf(PGMF,"# fortime %s\n",dates[Ti]); 
    fprintf(PGMF,"# param Height\n");
    fprintf(PGMF,"# isotherm %d\n",isotherm);
    fprintf(PGMF,"# projection radar {\n");
    fprintf(PGMF,"# type stereographic\n");
    fprintf(PGMF,"# centrallongitude %d\n",lon_0);
    fprintf(PGMF,"# centrallatitude %d\n",lat_0);
    fprintf(PGMF,"# truelatitude %d\n",lat_ts);
    fprintf(PGMF,"# bottomleft %f %f\n",SWlon,SWlat);
    fprintf(PGMF,"# topright %f %f\n",NElon,NElat);
    fprintf(PGMF,"# }\n");
    fprintf(PGMF,"%ld %ld\n255\n",outX,outY);
    for(ny=0;ny<outY;ny++)
      fwrite(&pgmarr[ny][0],outX,1,PGMF);
    fclose(PGMF);

    sprintf(levelname,"%s_%s_heights_per_radar.lis",dates[Ti],name);     
    HGTRADS=fopen(levelname,"w");
    for(rI=0;rI<RADN;rI++)
    {
      sprintf(radlevelname,"%s_%s_%sHEIGHT.txt",dates[Ti],radars[rI],name);
      RADHGT=fopen(radlevelname,"w");
      nx=radxy[rI][0];
      ny=radxy[rI][1];
      fprintf(HGTRADS,"%s %12ld %7.1f\n",radars[rI],T,intgrid[nx][ny]);
      fprintf(RADHGT,"%d",(int)((intgrid[nx][ny]+50.0)/100.0));
      fclose(RADHGT);
    } 
    fclose(HGTRADS);
  }

     for(t=0;t<Tdim;t++)
     {
        for(y=0;y<Ydim;y++) free(heights[t][y]);
	free(heights[t]);
     }
     free(heights);

  return(0);
}

/* ======================================================================== */

time_t sec_from_date(char *stim)
{
   struct tm Sd,*Sdd;
   int y,m;
   time_t secs;
  
   sscanf(stim,"%4d%2d%2d%2d%2d",
                   &y,&m,&Sd.tm_mday,&Sd.tm_hour,&Sd.tm_min);
   Sd.tm_year=y-1900;
   Sd.tm_mon=m-1;
   Sd.tm_sec=0;
   Sdd=&Sd;
  
   secs=mktime(Sdd);
   return(secs);
}

/*----------------------------------------------------------------------*/

void date_from_sec(char *datestr,time_t secs)
{
   struct tm Sdd;

   Sdd=*localtime(&secs);

   strftime(datestr,13,"%Y%m%d%H%M",&Sdd);
   return;
}
