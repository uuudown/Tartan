#========================================================================
#         Author:  Ang Li, PNNL
#        Website:  http://www.angliphd.com  
#        Created:  03/19/2018 04:05:39 PM, Richland, WA, USA.
#========================================================================


import string
import commands
import os
import math
import time

#============================= CONFIG ===============================
TIMES = 5
gpus = [1,2,4,8]
scale = ["strong","weak"]
schms = ["scale-out", "scale-out-pinned", "scale-out-rdma"]
#====================================================================

apps = []
#============================== APP =================================
b2reqwp = ["b2reqwp", "BRQ"]
diffusion = ["diffusion", "DFF"]
lulesh = ["lulesh", "LLH"]
comd = ["comd", "CMD"]
prbench = ["prbench", "PRB"]
hit = ["hit", "HIT"]
matvec = ["matvec", "MAM"]

apps.append(b2reqwp)
apps.append(diffusion)
apps.append(lulesh)
apps.append(comd)
apps.append(prbench)
apps.append(hit)
apps.append(matvec)
#====================================================================

def modify_lsf(nodes, scale, enablePAMI):
    lsf_file = open("exe.lsf","w")
    lsf_file.write("#!/bin/bash\n")
    lsf_file.write("#BSUB -P $YOUR PROJECTNO \n")
    lsf_file.write("#BSUB -W 10\n")
    lsf_file.write(str("#BSUB -nnodes ") + str(nodes) + "\n")
    lsf_file.write("module load gcc/5.4.0\n")
    lsf_file.write("module load hdf5\n")
    lsf_file.write("module load cuda/8.0.54\n")
    lsf_file.write("module load spectrum-mpi\n")
    #===========================
    if enablePAMI:
        lsf_file.write("source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu\n")
    else:
        lsf_file.write("export OMPI_MCA_pml_pami_enable_cuda=0 \n")
    #===========================

    cmd = str("/usr/bin/time -f 'ExE_Time: %e' jsrun -n") + str(nodes)\
            + " -a1 -g1 -c1 -r1 ./run_" + str(nodes) + "g_" + scale + ".sh"
    print cmd
    for i in range(0,TIMES):
        lsf_file.write(str(cmd) + "\n")
    lsf_file.close()

 
def run_one_app(app, outfile, schm, enablePAMI):
    os.chdir(app[0])
    os.system("make clean")
    os.system("make")

    for s in scale:
        for g in gpus:
            #we need to specially handle hit for weak scaling
            if s == "weak" and app[0] == "hit":
                mf_in = open("Makefile","r")
                mf_out = open("Makefile.tmp","w")
                for l in mf_in.readlines():
                    if l.startswith("SIZE"):
                        mf_out.write(str("SIZE = -D NSS=") + str(g*32) + "\n")
                    else:
                        mf_out.write(l)
                mf_in.close()
                mf_out.close()
                os.system("make clean")
                os.system("make -f Makefile.tmp")

            #generate lsf
            modify_lsf(g,s,enablePAMI)
            #clean output
            if os.path.exists("./myoutput"):
                    os.system("rm ./myoutput")
            #run
            cmmd = "bsub -o myoutput exe.lsf"
            print str("Run: ") + cmmd
            os.system(cmmd)
            ####whether finished
            while not os.path.exists("./myoutput"):
                time.sleep(1)

            print str("Finished ") +str(schm) + "_pami_" + str(enablePAMI) + " "\
                    + str(app[0]) + " " + str(s) + " with " + str(g) + " nodes."
            ####wait for data to write back
            time.sleep(80)

            ####process output
            ifile = open("./myoutput","r")
            feedback = ifile.readlines()
            ifile.close()

            all_time = 0.0
            for line in feedback:
                l = line.strip()
                if l.startswith("ExE_Time"):
                    one_time = float(l.split()[1])
                    all_time += one_time
            ####avg time
            all_time /= TIMES 
            print str(schm) + "_pami_" + str(enablePAMI) + " "  + str(app[0]) \
                    + " " + str(s) + " with " + str(g) +\
                    " nodes time is: " + str(all_time)
     
            outfile.write(str(schm)+"_" + str(enablePAMI) + ','\
                    + str(app[0]) + ',' + str(app[1]) + ',' + str(s) + ',' + str(g) + ','
                    + str(all_time) + '\n')

    os.chdir("..")


for schm in schms:
    if schm == "scale-out" or schm == "scale-out-pinned":
        #============ NO PAMI ================
        #outfile_name = str("res_") + str(schm) + ".txt"
        #outfile_path = str("./result/") + outfile_name
        #outfile = open(outfile_path, "w")
        #os.chdir(schm)
        #for app in apps:
            #run_one_app(app, outfile, schm, False)
        #os.chdir("..")
        #outfile.close()

        #============ PAMI ================
        outfile_name = str("res_") + str(schm) + "_pami.txt"
        outfile_path = str("./result/") + outfile_name
        outfile = open(outfile_path, "w")
        os.chdir(schm)
        for app in apps:
            run_one_app(app, outfile, schm, True)
        os.chdir("..")
        outfile.close()
    else:
        #============ RDMA ================
        outfile_name = str("res_") + str(schm) + ".txt"
        outfile_path = str("./result/") + outfile_name
        outfile = open(outfile_path, "w")
        os.chdir(schm)
        for app in apps:
            run_one_app(app, outfile, schm, True)
        os.chdir("..")
        outfile.close()


                


