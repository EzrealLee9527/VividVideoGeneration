import os
def volces_get_worker_cnt_worldsize():
    
    
    if os.environ.get("RLAUNCH_REPLICA") and os.environ.get("RLAUNCH_REPLICA_TOTAL"):
        print("Detect rlaunch replica mode")
        return int(os.environ.get("RLAUNCH_REPLICA")), int(os.environ.get("RLAUNCH_REPLICA_TOTAL"))
    
    hostname = os.environ.get("HOSTNAME",None) # only work for volces
    if 'worker' not in hostname: # hacky code
        hostname = None
    worldsize = int(os.environ.get("WORLD",1))
    worker_cnt = int(os.environ.get("RANK",0))
    if hostname is not None and worldsize==1:
        return int(hostname.split('-')[-1]), int(os.environ.get("MLP_WORKER_NUM",1))
    else:
        worldsize = int(os.environ.get("WORLD",1))
        worker_cnt = int(os.environ.get("RANK",0))
        return worker_cnt,worldsize