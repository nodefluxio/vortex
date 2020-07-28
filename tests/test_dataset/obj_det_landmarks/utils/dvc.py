import os
import sys
import subprocess
from typing import Union

def dvc_checkout(workdir : Union[str,None]=None) :
    if not workdir :
        workdir = os.environ['PWD']
    return subprocess.run(["dvc", "checkout"], check=True, stdout=subprocess.PIPE, cwd=workdir)

def dvc_push(remote : Union[str,None]=None, workdir : Union[str,None]=None, credential : str='gcs_credential.json', check=False) :
    """Push current dvc changes to given remote

    Parameters:
    remote : dvc remote, e.g. 'gcs'
    workdir : optional working directory, in which `dvc pull` should be called from
    credential : google application credentials, default to `gcs_credential.json`

    Returns:
    subprocess.CompletedProcess

    Raises:
    subprcess.CalledProcessError : if dvc push returns nonzero

    """
    cmd = ["dvc", "push"]
    if not remote is None :
        cmd.append("-r")
        cmd.append(remote)
    if not workdir :
        workdir = os.environ['PWD']
    env = os.environ.copy()
    env['GOOGLE_APPLICATION_CREDENTIALS'] = credential
    try :
        result = subprocess.run(cmd, check=check, stderr=subprocess.STDOUT, stdout=subprocess.STDOUT, env=env, cwd=workdir)
    except :
        result = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, env=env, cwd=workdir)
    return result

def dvc_pull(remote : Union[str,None]=None, workdir : Union[str,None]=None, credential : str='gcs_credential.json', check=False) :
    """Pull current dvc config from given remote

    Parameters:
    remote : dvc remote, e.g. 'gcs'
    workdir : optional working directory, in which `dvc pull` should be called from
    credential : google application credentials, default to `gcs_credential.json`

    Returns:
    subprocess.CompletedProcess

    Raises:
    subprcess.CalledProcessError : if dvc pull returns nonzero

    """
    cmd = ["dvc", "pull"]
    if not remote is None :
        cmd.append("-r")
        cmd.append(remote)
    if not workdir :
        workdir = os.environ['PWD']
    env = os.environ.copy()
    env['GOOGLE_APPLICATION_CREDENTIALS'] = credential
    try :
        result = subprocess.run(cmd, check=check, stderr=subprocess.STDOUT, stdout=subprocess.STDOUT, env=env, cwd=workdir)
    except :
        result = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, env=env, cwd=workdir)
    return result