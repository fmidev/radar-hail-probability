#include <ctype.h>                             
#include <locale.h>                            
#include <math.h>                              
#include <stdio.h>                             
#include <stdlib.h>                     
#include <string.h>                            

#include "sigtypes.h"
#include "headers.h" 
#include "product.h" 
#include "user_lib.h"
#include "dsp_lib.h" 


/* Function prototypes used just within this file
 */                                              
static FILE *ZEROHF;                             
static UINT1 *zerolevel;                         
static int Xdim,Ydim;                            
static char *POHstr,*T0str=NULL,POHname[13]={0},T0name[13]={0};
static void make_POH(const struct product_hdr *pin_head,       
                     struct product_hdr *pout_head );           
static void make_LHI(const struct product_hdr *pin_head,           
                     struct product_hdr *pout_head );               
static void make_flashes(const struct product_hdr *pin_head,   
                         struct product_hdr *pout_head );           
static void make_zeroheight(const struct product_hdr *pin_head,
                            struct product_hdr *pout_head );   

/* ====================================================================
 * This is a program to print out product data.                        
 */                                                                    
int main( int argc, char *argv[] )                                     
{                                                                      
  MESSAGE istatus;                                                     
  int i;                                                               
  struct product_hdr *pin_head, *pout_head;                            
  SINT4 in_size, in_channel, iout_size, iout_channel;                  
  char sin_path[160], sout_path[160],zeropath[160],                    
       pgmline[200],inname[200],*inputdir;                             

  /* -------------------------------------------------------------------- */
  setlocale( LC_ALL, "" );                                                  

  strcpy( sin_path, argv[1] );
  strcpy( zeropath, argv[2] );
  inputdir=argv[3];           
  POHstr=argv[4];             
  T0str=argv[5];              

  sprintf(POHname,"%-13s",POHstr);
  if(T0str) sprintf(T0name,"%-13s",T0str);


  istatus = imapopen( sin_path, 0, (void **)&pin_head, &in_size, &in_channel );
  if ( istatus != SS_NORMAL )                                                  
    {                                                                          
      sig_signal( istatus );                                                   
      exit(EXIT_FAILURE);                                                      
    }                                                                          

  /* First check that the data type is reflectivity: */
  switch( pin_head->pcf.idata_out )                    
    {                                                  
    case DB_HEIGHT: /* Hail probability based on TOP and zero level heights */
      ZEROHF=fopen(zeropath,"r");                                             
      for(i=0;i<3;i++)                                                        
      {                                                                       
         memset(pgmline,0,200);                                               
         fgets(pgmline,199,ZEROHF);                                           
         if(pgmline[0]=='#') { i--; continue; }                               
         if(i==1) sscanf(pgmline,"%d %d",&Xdim,&Ydim);                        
      }                                                                       
      printf("%d %d\n",Xdim,Ydim);                                            
      zerolevel=malloc(Xdim*Ydim);                                            
      fread(zerolevel,Xdim*Ydim,1,ZEROHF);                                    
      sprintf(sout_path,"%s/.%s_POH",inputdir,sin_path);                      
      istatus = imapcreate( sout_path, in_size, (void **)&pout_head,          
                            &iout_size, &iout_channel );                      
      if ( istatus != SS_NORMAL )                                             
        sig_signal( istatus );                                                
      else                                                                    
        {                                                                     
          make_POH( pin_head, pout_head );                                    
          istatus = imapclose( pout_head, iout_size, iout_channel );          
          sprintf(inname,"%s/%s_POH",inputdir,sin_path);                      
          rename(sout_path,inname);                                           
        }                                                                     

      if(T0str)
      {        
        printf("Generating zero level data\n");
      /* Replace original TOP data by zero level height */
      sprintf(sout_path,"%s/.%s_T0",inputdir,sin_path);   
      istatus = imapcreate( sout_path, in_size, (void **)&pout_head,
                            &iout_size, &iout_channel );            
      if ( istatus != SS_NORMAL )                                   
        sig_signal( istatus );                                      
      else                                                          
        {                                                           
          make_zeroheight( pin_head, pout_head );                   
          istatus = imapclose( pout_head, iout_size, iout_channel );
          sprintf(inname,"%s/%s_T0",inputdir,sin_path);             
          rename(sout_path,inname);                                 
        }                                                           
      }                                                             
      free(zerolevel);                                              
      fclose(ZEROHF);                                               

      break;

    case DB_DBZ: /* Flashes marked as dBZ>100, 5 min accumulation */
      istatus = imapcreate( sout_path, in_size, (void **)&pout_head,
                            &iout_size, &iout_channel );            
      if ( istatus != SS_NORMAL )                                   
        sig_signal( istatus );                                      
      else                                                          
        {                                                           
          make_flashes( pin_head, pout_head );                      
          istatus = imapclose( pout_head, iout_size, iout_channel );
        }                                                           
      break;                                                        

    default:
      printf("Sorry, change_product will not work on %d data.\n", pin_head->pcf.idata_out);
      break;                                                                               
    }                                                                                      

  istatus = imapclose( pin_head, in_size, in_channel );
                                                       
  exit(EXIT_SUCCESS);                                  
}                                                      

/* ============================================
 * Function to convert data to the new file.   
 */                                            
static void make_POH                           
(const struct product_hdr *pin_head,           
 struct product_hdr *pout_head )               
{                                              
  const UINT1 *pin_data;                       
  UINT1 *pout_data,cPOH;                       
  SINT4 isize, ix, iy,X,Y,zeroH,TOPH,N;        
  FLT4 POH,dH;                                 

  /* First copy accross the headers and data */
  pin_data  = sizeof(struct product_hdr) + (const UINT1 *)pin_head;
  pout_data = sizeof(struct product_hdr) + (UINT1 *)pout_head;     

  *pout_head = *pin_head;

  isize = pin_head->pcf.ixsize*pin_head->pcf.iysize;

  memmove( pout_data, pin_data, isize );
  X=pin_head->pcf.ixsize;               
  Y=pin_head->pcf.iysize;               

  /* Substract zero level height from TOP 45 dBZ height to get dH
     for calculation of POH */                                   

  printf("%ld %ld\n\n",X,Y);
  for (iy=0; iy<Y; iy++)    
  {                         
      for (ix=0; ix<X; ix++)
      {                     
          N=X*iy+ix;        
          TOPH=pout_data[N];
          if(TOPH==255 || TOPH==0) continue;
          if(TOPH!=254)                     
          {                                 
             zeroH=zerolevel[N];            
             dH=(TOPH-zeroH)/10.0;          
             POH=0.319+0.133*dH;            
             if(POH<0.0) POH=0.0;           
             if(POH>1.0) POH=1.0;           
                                            
             printf("%ld %ld %ld %ld %.1f %.2f\n",ix,iy,TOPH,zeroH,dH,POH);
                                                                           
             cPOH=(unsigned char)(100.0*POH)+1;                            
          } else cPOH = 254;                                               
          pout_data[N]=cPOH;                                               
      }                                                                    
      /*      if(iy>500 && iy<950) pout_data[X*iy+500]=100; */             
   }                                                                       
  /* Change the product name */                                            
  memmove( pout_head->pcf.spname,POHname,12);                              
  /* memmove( &pout_head->pcf.psi[0], "P of hail        ", 17);            
  pout_head->pcf.iptype=PROD_USER;                                         
  pout_head->pcf.idata_out=DB_USER;                                        
  */                                                                       
  return;                                                                  
}                                                                          

/* ============================================
 * Function to generate large hail index   
 */                                            
static void make_LHI                           
(const struct product_hdr *pin_head,           
 struct product_hdr *pout_head )               
{                                              
  const UINT1 *pin_data;                       
  UINT1 *pout_data,cLHI;                       
  SINT4 isize, ix, iy,X,Y,M20H,TOPH,N;        
  FLT4 LHI,dH;                                 

  /* First copy accross the headers and data */
  pin_data  = sizeof(struct product_hdr) + (const UINT1 *)pin_head;
  pout_data = sizeof(struct product_hdr) + (UINT1 *)pout_head;     

  *pout_head = *pin_head;

  isize = pin_head->pcf.ixsize*pin_head->pcf.iysize;

  memmove( pout_data, pin_data, isize );
  X=pin_head->pcf.ixsize;               
  Y=pin_head->pcf.iysize;               

  /* Substract zero level height from TOP 45 dBZ height to get dH
     for calculation of LHI */                                   

  printf("%ld %ld\n\n",X,Y);
  for (iy=0; iy<Y; iy++)    
  {                         
      for (ix=0; ix<X; ix++)
      {                     
          N=X*iy+ix;        
          TOPH=pout_data[N];
          if(TOPH==255 || TOPH==0) continue;
          if(TOPH!=254)                     
          {                                 
             M20H=zerolevel[N];            
             dH=(TOPH-M20H)/10.0;          
             LHI=100.0+dH;            
             if(LHI<0.0) LHI=0.0;           
             if(LHI>253) LHI=253;           
                                            
             printf("%ld %ld %ld %ld %.1f %.2f\n",ix,iy,TOPH,M20H,dH,LHI);
                                                                           
             cLHI=(unsigned char)LHI+1;                            
          } else cLHI = 254;                                               
          pout_data[N]=cLHI;                                               
      }                                                                    
      /*      if(iy>500 && iy<950) pout_data[X*iy+500]=100; */             
   }                                                                       
  /* Change the product name */                                            
  memmove( pout_head->pcf.spname,POHname,12);                              
  /* memmove( &pout_head->pcf.psi[0], "P of hail        ", 17);            
  pout_head->pcf.iptype=PROD_USER;                                         
  pout_head->pcf.idata_out=DB_USER;                                        
  */                                                                       
  return;                                                                  
}                                                                          




/* ===========================================================
 * Function to make zerolevel height product from TOP product 
 */                                                           
static void make_zeroheight                                   
(const struct product_hdr *pin_head,                          
 struct product_hdr *pout_head )                              
{                                                             
  const UINT1 *pin_data;                                      
  UINT1 *pout_data,cPOH;                                      
  SINT4 isize, ix, iy,X,Y,zeroH,TOPH,N;                       
  FLT4 POH,dH;                                                

  /* First copy accross the headers and data */
  pin_data  = sizeof(struct product_hdr) + (const UINT1 *)pin_head;
  pout_data = sizeof(struct product_hdr) + (UINT1 *)pout_head;     

  *pout_head = *pin_head;

  isize = pin_head->pcf.ixsize*pin_head->pcf.iysize;

  memmove( pout_data, pin_data, isize );
  X=pin_head->pcf.ixsize;
  Y=pin_head->pcf.iysize;

  /* Substract zero level height from TOP 45 dBZ height to get dH
     for calculation of POH */

  printf("%ld %ld\n\n",X,Y);
  for (iy=0; iy<Y; iy++)
  {
      for (ix=0; ix<X; ix++)
      {
          N=X*iy+ix;
          zeroH=zerolevel[N];
          pout_data[N]=(UINT1)zeroH;
      }
   }
  /* Change the product name */
  memmove( pout_head->pcf.spname,T0name,12);
  return;
}


static void make_flashes(const struct product_hdr *pin_head,
                    struct product_hdr *pout_head )
{
}

