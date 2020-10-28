# Run an Experiment on AdaptDL

===

## Introduction to AdaptDL

AdaptDL is a resource-adaptive framework for distributed deep learning training. Not only to support the distributed deep learning training, it can also do elastic job scheduling and training process acceleration, such learning-rate adjustment. More detailed information about AdaptDl can be found [here](https://adaptdl.readthedocs.io/en/latest/)

## Prerequisites

To enable the AdaptDL backend for NNI, users need to have a [Kubernetes Cluster](https://kubernetes.io/docs/setup/) and [install AdaptDL](https://adaptdl.readthedocs.io/en/latest/installation/index.html) on their K8S cluster.

## Execute AdaptDL from NNI

Just as other NNI experiments, users just need to provide a experiment configuration file and a hyper-parameter json file. Here is an example of the configuration file.

```yaml
authorName: default
experimentName: cifar_adl
trialConcurrency: 1
maxExecDuration: 3h
maxTrialNum: 10
nniManagerIp: 10.20.41.10
trainingServicePlatform: adl
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: minimize
trial:
  command: python3 /main_nni.py
  codeDir: .
  gpuNum: 1
  image: registry.petuum.com/dev/nni:adaptdl
  imagePullSecrets:
    - name: imagepullsecret
  adaptive: true
  checkpoint:
    storageClass: dfs
    storageSize: 1Gi
  nfs:
    server: 172.20.188.236
    path: /
    containerMountPath: /nfs
```

Training codes are assumed to be with the given docker image. For now, the users have to create the images by themselves.

### Shared Volumes in AdaptDL jobs

A [Trial](https://nni.readthedocs.io/en/latest/TrialExample/Trials.html) in NNI is an individual attempt at applying a configuration (e.g., a set of hyper-parameters) to a model. So, the AdaptDL job under each trial are also independent. However, the persistence volumes are applied for each AdaptDl job to share data. Users can set up the shared volume in the `checkpoint` field.

`ADAPTDL_CHECKPOINT_PATH` and `ADAPTDL_SHARE_PATH` are two environment variables indicate two persistent volumes paths, `/adaptdl/checkpoint` and `/adaptdl/share`.
`ADAPTDL_CHECKPOINT_PATH` is used by AdaptDL job rescheduling. When AdaptDL scheduler reschedules the training job resource, training pods will be restarted. To keep the training consistent, AdaptDL saves the training states to the checkpoints under this path. 
`ADAPTDL_SHARE_PATH` is created for users. One example usage is save downloaded training data to reduce duplicate downloading.
The life-cycle of these two persistent volumes, `/adaptdl/checkpoint` and `/adaptdl/share`, are as the same as the AdaptDL jobs. For now, the AdaptDL jobs will only be deleted after the experiment is done for developing purpose, which means the users can still check the error messages after the jobs failure.

There is another persistent volume `/adaptdl/tensorboard` which can get from the environment variable `ADAPTDLCTL_TENSORBOARD_LOGDIR`. This space is created for saving the TensorBoard checkpoints. The life-cycle of `/adaptdl/tensorboard` is as the same as the experiment. So, even after the AdaptDL job is done, it still can be accessed by users.

### NFS in AdaptDL jobs

Our design also supports direct save and load through NFS. `server`: NFS server address, e.g. IP address or domain; `path`: NFS server export path, i.e. the absolute path in NFS that can be mounted to trials; `containerMountPath`: In container absolute path to mount the NFS path above.
Unlike shared volumes, NFS can share data between different trials and experiments.

### Training code execution in AdaptDL jobs

To run the NNI experiment with AdaptDL, users should provide a docker image which contains the training code and other artifacts. Meanwhile, on NNI side, it will provide 2 scripts. One is for training execution, another is for clean up. A K8S configMap is used to attach these 2 scripts into each AdaptDL job. The life-cycle of the configMap is as the same as the corresponding AdaptDL job. Here are some details about these 2 scripts.

`run.sh` is the script to setup necessary environment variables used by NNI, such as trial id, NNI platform name. At the same time, it will call [trial_keeper] to trigger the real training code in a subprocess.

`cleanup.sh` is the script to clean up the AdaptDL pods with correct signal handling. AdaptDL has a specific signal handler to save the states between each pods restart. However, as we know above, there are many execution process layers between `run.sh` and the real training command. In this case, `cleanup.sh` guarantees the AdaptDL signal handling will work as expected.

### NNI Trial Status and Training Information

NNI trial status are related to the corresponding AdaptDL job status, including `WAITING`, `RUNNING`, `FAILURE` and `SUCCEED`. `WAITING` means the job is in preparation. `RUNNING` means the job is in training, `FAILURE` and `SUCCEED` indicates the final status of the job.
Once the AdaptDL job is done or failed, the status will be updated to the trial status and can be checked through `nnictl` or NNI web UI. Also, the detailed training information can be found in the trial messages, which makes it easier for users to debug.