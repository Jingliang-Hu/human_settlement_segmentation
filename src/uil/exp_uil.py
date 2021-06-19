from datetime import datetime
import os


def initialOutputFolder(paraDict):
    # get time stamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    outcomeDir = paraDict["backbone_model"] + '_' + paraDict["solution"] + '_'+ paraDict["exper_description"] + '_outcome_' + timestamp
    # mkdir output folder
    os.mkdir(outcomeDir)
    return outcomeDir

def recordExpParameters(outcomeDir,paraDict):
    # record parameters
    f = open(os.path.join(outcomeDir,'parameters.txt'),"w")
    f.write("PARAMETERS:")
    f.write("\n")
    for key,value in paraDict.items():
        try:
            f.write(key+':'+value)
        except:
            f.write(key+':'+str(value))
        f.write("\n")
    f.close()
    del f
    return 0



