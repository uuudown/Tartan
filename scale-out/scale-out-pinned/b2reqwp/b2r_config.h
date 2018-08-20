#ifndef _B2R_CONFIG_
#define _B2R_CONFIG_

// simulation environment size, support 2D or 3D, square/rectangle or square/cuboid 

#define ENV_DIM_X           500
#define ENV_DIM_Y           40
#define ENV_DIM_Z           500

/*#define ENV_DIM_X           3000*/
/*#define ENV_DIM_Y           200*/
/*#define ENV_DIM_Z           3000*/


#define GBL_ENV             ((ENV_DIM_X)*(ENV_DIM_Y)*(ENV_DIM_Z))
#define B2R_D               2       // must be: B2R_D * B2R_R < B/2

#define _PERIODIC_          0       // set as 0 to disbale periodic process
#define _ENV_3D_            1       // 1 -> 3D simulation environment, 0 -> 2D environment
#define _DEBUG_             0       // 1 -> enable debug mode, the block data will be print (to file) 0 -> only print execution time 
#define _MDL_DEV_           1       // device for executing the model, 0 -> CPU, 1 -> GPU

typedef float CELL_DT;

#endif
