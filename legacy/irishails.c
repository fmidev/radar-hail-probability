/* Calculates hail indices HHI, THI and LHI (arg 4) from height difference between 
   IRIS TOP (arg 1) and isotherm height (arg 2)
   Both input files must be in 1-byte IRIS TOP composite format. 

   Argument 3 is output file also in 1-byte IRIS CAPPI composite format. The TOP input file acts as a template
   for output file. Product name and type are changed in header information, and CAPPI specific info
   bytes are set to zero.

   Harri Hohti FMI / 20190603

*/

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <stdint.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <unistd.h>
# define TOPS 0
# define ISOT 1
# define OUT 2
# define HDRLEN 640

double Holleman(double dH);

int main(int argc, char *argv[])
{
  FILE *INF, *OUTF;
  uint8_t hdr[2][HDRLEN], outhdr[HDRLEN], *data[3],ind,namespace[12];
  size_t datasize;
  int i,exitcode=0,dBI;
  double isom,topm;
  uint8_t LHI=0,THI=0,HHI=0,dBN,isoval,topval;
  long N;
  int prodoffs=24, nameoffs=74, specoffs=176, typeoffs=142, typeoffs_in=156;
  uint16_t prodcode=3; /* IRIS CAPPI */
  uint16_t typecode=2; /* DB_DBZ */
  char *product,*outfile,prodname[13]={0};
  double POH,dB;
  double dH; /* height diffrence [m] */
  struct stat iristat;

  stat(argv[1],&iristat);
  datasize=iristat.st_size-HDRLEN;

  for(i=0;i<3;i++)
  {
     data[i]=malloc(datasize);
     if(i<2)
     {
        INF=fopen(argv[i+1],"r");
        fread(hdr[i],1,HDRLEN,INF);
        fread(data[i],1,datasize,INF);
        fclose(INF);
     }
  }

  memcpy(outhdr,hdr[TOPS],HDRLEN);  
  outfile=argv[3];
  product=argv[4];

  if(strstr(product,"LHI")) LHI=1;
  if(strstr(product,"THI")) THI=1;
  if(strstr(product,"HHI")) HHI=1;

  /*   printf("%s %d %d %d\n",product,LHI,HHI,THI); */

  if(!(LHI | HHI | THI))
  {
     printf("Give product as LHI, HHI or THI!\n");
     exitcode=1;
  }
  else
  {
     sprintf(prodname,"HAIL_%s",product);
     memset(namespace,32,12);
     memcpy(namespace,prodname,strlen(prodname));
     OUTF=fopen(outfile,"w");

     for(N=0;N<datasize;N++)
     {
        topval=data[TOPS][N];
        topm=((double)topval-1.0)*100.0;
        isoval=data[ISOT][N];
        isom=((double)isoval-1.0)*100.0;
        dH=topm-isom; /* height diffrence [m] */

        dBN=topval;
        while(topval && topval<=254)
        {
           if(LHI)
           {
             dB = dH/1000.0; /* [km] */
           }

           if(HHI)
           {
 	     POH = Holleman(dH);
             dB=10.0*POH;
             if(dB<0.0) dB=0.0;
           }

           if(THI)
           {
	     POH = Holleman(dH);
             dB=10.0*POH;
             while(1)
	     {
                if(isom<=1200) { dB+=2.0; break; }
                if(isom<=1700) { dB+=1.0; break; }
                if(isom<=3500) break;
                dB-=1.0;
                break;
	     }
             if(dB<0.0) dB=0.0;
           }

      	   dBI = (int)(2.0*dB+64.0);
           dBN=(uint8_t)dBI;
           if(dBI<0) dBN=0;
           if(dBI>=253) dBN=253;

           break;
	}
        if(topval==254) dBN=0;
        data[OUT][N]=dBN;
 
     }
     /* IRIS header modifications to hdr[TMPL] here */
     memset(outhdr+specoffs,0,80);
     memcpy(outhdr+prodoffs,&prodcode,2);
     memcpy(outhdr+typeoffs,&typecode,2);
     memcpy(outhdr+typeoffs_in,&typecode,2);
     memcpy(outhdr+nameoffs,namespace,12);
     /*---------------- */

     fwrite(outhdr,1,HDRLEN,OUTF);
     fwrite(data[OUT],1,datasize,OUTF);
     fclose(OUTF);
  }

  for(i=0;i<3;i++) free(data[i]);

  return(exitcode);
  
}

double Holleman(double dH)
{
  return(0.319 + 0.000133*dH);  
}
