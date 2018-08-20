#========================================================================
#         Author:  Ang Li, PNNL
#        Website:  http://www.angliphd.com  
#        Created:  03/19/2018 03:09:45 PM, Richland, WA, USA.
#========================================================================

import string
import commands
import os
import math

#============================= CONFIG ===============================
TIMES = 5
gpus = [1,2,4,8]
scale = ["strong","weak"]
schms = ["scale-up", "scale-up-nvlink"]
#====================================================================

apps = []
#============================== APP =================================
convnet2 = ["convnet2", "CNN"]
cusimann = ["cusimann", "CSM"]
gmm = ["gmm", "GMM"]
kmeans = ["kmeans", "KMN"]
montercarlo = ["montecarlo", "MTC"]
planar = ["planar", "PLN"]
trueke = ["trueke", "TRK"]

apps.append(convnet2)
apps.append(cusimann)
apps.append(gmm)
apps.append(kmeans)
apps.append(montercarlo)
apps.append(planar)
apps.append(trueke)
#====================================================================

def run_one_app(app, outfile, schm):
    os.chdir(app[0])
    os.system("make clean")
    os.system("make")

    for s in scale:
        for g in gpus:
            cmd = str("/usr/bin/time -f '%e' ./run_") + str(g) + str("g_") + str(s) + ".sh"
            print str('$Run ') + app[0] + ':' + cmd
            time = 0.0
            for t in range(0,TIMES):
                time += float(commands.getoutput(cmd).split('\n')[-1])
            time /= TIMES
            line = str(schm) + "," + str(app[1]) + "," + str(s) + "," + str(g) + "," + str(time)
            print line
            outfile.write(line + "\n")

    os.chdir("..")


for schm in schms:
    outfile_name = str("res_") + str(schm) + ".txt"
    outfile_path = str("./result/") + outfile_name
    outfile = open(outfile_path, "w")

    os.chdir(schm)
    for app in apps:
        run_one_app(app, outfile, schm)
        
    os.chdir("..")
    outfile.close()
