import string
import commands
import os
import math
import time

# ====================================== Configuration ========================================
#
#primitives = ['all_gather_perf', 'all_reduce_perf', 'broadcast_perf','reduce_perf','reduce_scatter_perf']
#start = '8'
#end = '1GB'

primitives = ['reduce_scatter_perf']
start = '64MB'
end = '64MB'

def modify_lsf(op, nodes, start, end):
    lsf_file = open("exe.lsf","w")
    lsf_file.write("#!/bin/bash\n")
    lsf_file.write("#BSUB -P YOUR_PROJECT_ID\n")
    lsf_file.write("#BSUB -W 10\n")
    lsf_file.write(str("#BSUB -nnodes ") + str(nodes) + "\n")
    lsf_file.write("module load cuda/8.0.54\n")
    lsf_file.write("module load spectrum-mpi\n")
    #===========================
    lsf_file.write("source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu\n")
    #lsf_file.write("export OMPI_MCA_pml_pami_enable_cuda=0 \n")
    #===========================

    cmd = str("jsrun -n") + str(nodes) + " -a1 -g1 -c1 -r1 ./" + str(op) + ' -b ' + str(start) + ' -e ' + str(end) + ' -f 2 -g 1 -c 0'
    print cmd
    lsf_file.write(str(cmd) + "\n")
    lsf_file.close()
        
def latency_test(op, outfile):
    for i in range(2,9):
        modify_lsf(op, i, start, start)
        if os.path.exists("./output"):
                os.system("rm ./output")
        cmmd = "bsub -o output exe.lsf"
        print str("Run: ") + cmmd
        os.system(cmmd)
        while not os.path.exists("./output"):
            time.sleep(1)
        print str("Finished ") + str(op) + " for latency with " + str(i) + " nodes."

        time.sleep(80)
        ifile = open("./output","r")
        feedback = ifile.readlines()
        ifile.close()
        time_pos = -1
        bw_pos = -1
        for line in feedback:
            l = line.strip()
            if l.find('algbw') != -1:
                values = l[1:].strip().split()
                print values
                for j in range(0,len(values)):
                    if values[j] == 'time':
                        time_pos = j
                        break
            if len(l) >0 and l[0].isdigit():
                values = l.split()
                ttime = (values[time_pos])
                print str('OP ') + op + ' by ' + str(i) + " GPU CL Time is " + str(ttime) + ".\n"
                outfile.write(str(op) + ',' + str(i) + ',' + str(ttime) + '\n')

def bandwidth_test(op, outfile):
    for i in range(2,9):
        modify_lsf(op, i, end, end)
        if os.path.exists("./output"):
                os.system("rm ./output")
        cmmd = "bsub -o output exe.lsf"
        print str("Run: ") + cmmd
        os.system(cmmd)
        while not os.path.exists("./output"):
            time.sleep(1)
        print str("Finished ") + str(op) + " for bandwidth with " + str(i) + " nodes."

        time.sleep(80)
        ifile = open("./output","r")
        feedback = ifile.readlines()
        ifile.close()

        time_pos = -1
        bw_pos = -1
        for line in feedback:
            l = line.strip()
            if l.find('algbw') != -1:
                print l
                values = l[1:].strip().split()
                for j in range(0,len(values)):
                    if values[j] == 'busbw':
                        bw_pos = j
                        break
            if len(l) > 0 and l[0].isdigit():
                values = l.split()
                print values
                bw = (values[bw_pos])
                print str('OP ') + op + ' by ' + str(i) + "GPU CL BW is " + str(bw) + ".\n"
                outfile.write(str(op) + ',' + str(i) + ',' + str(bw) + '\n')

def packet_test(op, outfile):
    modify_lsf(op, 8, start, end)
    if os.path.exists("./output"):
            os.system(str("mv ./output ./dump_packet_") + op)
    cmmd = "bsub -o output exe.lsf"
    print str("Run: ") + cmmd
    os.system(cmmd)
    while not os.path.exists("./output"):
        time.sleep(1)
    print str("Finished ") + str(op) + " for packet with 8 nodes."
    time.sleep(60)

    ifile = open("./output","r")
    feedback = ifile.readlines()
    ifile.close()

    time_pos = -1
    bw_pos = -1
    for line in feedback:
        l = line.strip()
        if l.find('algbw') != -1:
            values = l[1:].strip().split()
            print values
            for j in range(0,len(values)):
                if values[j] == 'busbw':
                    bw_pos = j
                    break
        if len(l)>0 and l[0].isdigit():
            values = l.split()
            print values
            bt = (values[0])
            bw = (values[bw_pos])
            outfile.write(str(op) + ',' + str(bt) + ',' + str(bw) + '\n')
    

#latency_file = open("latency.txt","w")
#latency_file.write("OP,GPU,Latency\n")
#bandwidth_file = open("bandwidth.txt","w")
#bandwidth_file.write("OP,GPU,Bandwidth\n")
packet_file = open("packet.txt","w")
packet_file.write("OP,Packet,Bandwidth\n")


for op in primitives:
    #latency_test(op, latency_file)
    #bandwidth_test(op, bandwidth_file)
    packet_test(op, packet_file)

#latency_file.close()
#bandwidth_file.close()
packet_file.close()

