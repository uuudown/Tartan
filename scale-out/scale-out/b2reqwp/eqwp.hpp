#ifndef _FDM3D_HPP_
#define _FDM3D_HPP_
#include <cmath>
using namespace std;

CELL_DT* pinned_alloc(int items);
void free_pinned_mem(CELL_DT *p);
void eqwp_cuda_init(int Ngx, int Ngy, int Ngz);
double eqwp_gpu_main(int ts, int b2r_R, int Ngx, int Ngy, int Ngz, bool op_flag);
/*
*********************************************************************
* func   name: gol_model_init
* description: this function initializes cells according to model
               requirements, this is to initialize from point view
               of the model, no B2R issue needs to be considered.
* parameters :
*             none
* return: none
*********************************************************************
*/
void model_init(int rank)
{

}

/*
*********************************************************************
* func   name: model_finalize
* description:   release some model scopic resources
* parameters :
*             none
* return: none
*********************************************************************
*/
void model_finalize()
{
    //free_pinned_mem();
}

/*
 *********************************************************************
 * func   name: gol_live_neighbor_cnt
 * description: Add up all live neighbors and return the number of 
 *              live neighbors
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */

int gol_live_neighbor_cnt(int x, int y, int j)
{
    return 0;
}

/*
 *********************************************************************
 * func   name: print_block_data
 * description: this function prints / output results in local blocks,
                for results verification
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */
void print_block_data(CELL_DT *p_grid)
{
    int global_id;
    int Rd = B2R_R*B2R_D;
    for (int z = Rd; z < B2R_BLOCK_SIZE_Z-Rd; ++z)
    {
        for(int r = Rd; r < B2R_BLOCK_SIZE_Y-Rd; r++)
        {
            for(int c = Rd; c < B2R_BLOCK_SIZE_X-Rd; c++)
            {
                global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                cout << p_grid[global_id] << ",";
            }
            cout << endl;
        }
    }
}
/*
 *********************************************************************
 * func   name: print_sent_data
 * description: mostly for debuging, this function prints / output,
                packed data in sending buffer
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */
void print_sent_data()
{
}
/*
 *********************************************************************
 * func   name: print_recv_data
 * description: mostly for debuging, this function prints / output,
                packed data in sending buffer
 * parameters:
 *             none
 * return: none
 *********************************************************************
 */
void print_recv_data()
{

}

#endif
