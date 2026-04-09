#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <projects.h>


/* static double rXdim=758896.89,rYdim=1225688.93; */
static double rXdim=814.24032,rYdim=1312.88766;
static double degree=111194.0;
static int XDIM=1628,YDIM=2625;

/* SUOMI500N -polaaristereografinen projektio */
static char *stereparams[]=
{
    "proj=stere",
    "a=6371000",
    "b=6371000",
    "lon_0=25E",
    "lat_0=90N",
    "lat_ts=90N",
    "x_0=408214.26",
    "y_0=3639312.10",
    "no_defs"
};

/* SW 57.93/18.6 NE 69.0/34.9 408214.26	3639312.10  */
 
int main(int argc, char *argv[])
{
    projUV stere,geo;
    PJ *steref;
    int i,N,X,Y,geoN,geosize,geoX,geoY,latdim,londim;
    FILE *STEREF, *GEOF;
    double kx,ky,lon,lat,latd,lond,startlat,stoplat,startlon,stoplon;  
    char sbuf[300];
    unsigned char *latlonpic,*sterepic;

    kx=(double)XDIM/rXdim;;
    ky=(double)YDIM/rYdim;;
    latd=500.0/degree;
    lond=latd*2.0;
    startlat=57.93+latd/2.0;
    stoplat=69.0-latd/2.0;
    startlon=18.6+lond/2.0;
    stoplon=34.9-lond/2.0;

    steref=pj_init(sizeof(stereparams)/sizeof(char *),stereparams);

    geo.v=startlat*DEG_TO_RAD;
    geo.u=startlon*DEG_TO_RAD;
    stere=pj_fwd(geo,steref);
    printf("%.0f %.0f\n",stere.u/1000.0,stere.v/1000.0);

    geo.v=stoplat*DEG_TO_RAD;
    geo.u=stoplon*DEG_TO_RAD;
    stere=pj_fwd(geo,steref);
    printf("%.0f %.0f\n",stere.u/1000.0,stere.v/1000.0);

    GEOF=fopen(argv[2],"w");
    STEREF=fopen(argv[1],"r");

    latdim=(int)((stoplat-startlat)/latd);
    londim=(int)((stoplon-startlon)/lond);
    printf("W=%f E=%f S=%f N=%f %d %d\n",startlon,stoplon,startlat,stoplat,londim,latdim);
    geosize=latdim*londim;
    latlonpic=calloc(geosize*3,1);
    sterepic=malloc(XDIM*YDIM*3);
    for(i=0;i<3;i++)
    {
      fgets(sbuf,299,STEREF);
      if(sbuf[0]=='#') {i--; continue; }
      printf("%s",sbuf);
    }
    fread(sterepic,XDIM*YDIM*3,1,STEREF);

    for(geoY=0;geoY<latdim;geoY++)
    {
       lat=startlat+geoY*latd;
       geo.v=lat*DEG_TO_RAD;
       for(geoX=0;geoX<londim;geoX++)
       {
         lon=startlon+geoX*lond;
         geoN=(geoY*londim+geoX)*3;
         geo.u=lon*DEG_TO_RAD;
         stere=pj_fwd(geo,steref);
         X=(int)(kx*stere.u/1000.0);
         Y=YDIM-(int)(ky*stere.v/1000.0)-1;
         if(X<0 || X>=XDIM || Y<0 || Y>=YDIM) continue;
         N=(Y*XDIM+X)*3;
         for(i=0;i<3;i++)
	 {
           latlonpic[geoN+i]=sterepic[N+i];
	 }
       }
    }
    printf("%.4f %.4f\n",lon,lat);

    fprintf(GEOF,"P6\n%d %d\n255\n",londim,latdim);
    fwrite(latlonpic,geosize*3,1,GEOF);
 
    fclose(GEOF);
    fclose(STEREF);

    free(sterepic);
    free(latlonpic);

  
    return(0);

}
