#==========================================================================================
# This microbenchmark belongs to the Tartan Benchmark Suite, which contains microbenchmarks
# and benchmark applications for evaluating multi-GPUs and multi-GPU interconnect. Please see
# our IISWC-18 paper titled "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU 
# Benchmark Suite" for detail. 
#
#        Version:  1.0
#        Created:  01/24/2018 03:52:11 PM
#
#         Author:  Ang Li, PNNL
#        Website:  http://www.angliphd.com  
#==========================================================================================

import string
import commands
import os
import math

# ====================================== Configuration ========================================

primitives = ['all_gather_perf', 'all_reduce_perf', 'broadcast_perf','reduce_perf','reduce_scatter_perf']
#primitives = ['reduce_scatter_perf']
start = '8'
end = '1GB'


def latency_test(op, outfile):
    commd = str("./")+ str(op) + ' -b ' + start + ' -e ' + start + ' -f 2 '
    for i in range(1,9):
        cmd = commd + " -g " + str(i)
        print cmd
        feedback = commands.getoutput(cmd).strip().split('\n')
        time_pos = -1
        bw_pos = -1
        for line in feedback:
            l = line.strip()
            if l.find('bytes') != -1:
                values = l[1:].strip().split()
                print values
                for j in range(0,len(values)):
                    if values[j] == 'time':
                        time_pos = j
                        break
            if len(l) >0 and l[0].isdigit():
                values = l.split()
                time = (values[time_pos])
                print str('OP ') + op + ' by ' + str(i) + " GPU CL Time is " + str(time) + ".\n"
                outfile.write(str(op) + ',' + str(i) + ',' + str(time) + '\n')

def bandwidth_test(op, outfile):
    commd = str("./")+ str(op) + ' -b ' + end + ' -e ' + end + ' -f 2 '
    for i in range(1,9):
        cmd = commd + " -g " + str(i)
        feedback = commands.getoutput(cmd).strip().split('\n')
        time_pos = -1
        bw_pos = -1
        for line in feedback:
            l = line.strip()
            if l.find('bytes') != -1:
                values = l[1:].strip().split()
                print values
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
    cmd = str("./")+ str(op) + ' -b ' + start + ' -e ' + end + ' -f 2 -g 8 '
    feedback = commands.getoutput(cmd).strip().split('\n')
    time_pos = -1
    bw_pos = -1
    for line in feedback:
        l = line.strip()
        if l.find('bytes') != -1:
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
    

latency_file = open("latency.txt","w")
latency_file.write("OP,GPU,Latency\n")
bandwidth_file = open("bandwidth.txt","w")
bandwidth_file.write("OP,GPU,Bandwidth\n")
packet_file = open("packet.txt","w")
packet_file.write("OP,Packet,Bandwidth\n")


for op in primitives:
    latency_test(op, latency_file)
    bandwidth_test(op, bandwidth_file)
    packet_test(op, packet_file)

latency_file.close()
bandwidth_file.close()
packet_file.close()

