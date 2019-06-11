# Miscellaneous Details

## SSH

#### Creating an SSH key:

```bash
ssh-keygen -t rsa -b 4096 -C "[email]"
```

#### Logging in via SSH:

```bash
ssh <user@remote>
```

#### Installing Anaconda3 via SSH:

Find the latest anaconda version on https://repo.continuum.io/archive/

```bash
wget -c https://repo.continuum.io/archive/Anaconda3-<version>-Linux-x86_64.sh
bash Anaconda3-<version>-Linux-x86_64.sh
```

#### Installing `tensorflow-gpu` via SSH (Linux):

Follow this: https://github.com/williamFalcon/tensorflow-gpu-install-ubuntu-16.04

This guide will install CUDA 9.0 with python 3.6.x, with tensorflow-gpu version <= 1.12.0.

- `pip install -U ray`
- `pip install serializable`
- mujoco py

#### View tensorboard results on localhost:

1. From the local machine:

   ```bash
   ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
   ```

2. From the remote machine:

   ```bash
   tensorboard --logdir <path> --port 6006
   ```

3. View locally at `localhost:16006`

   

## GPU Usage

#### Specify which GPU to train on:

```bash
export CUDA_VISIBLE_DEVICES=1,2 # This selects GPUs 1 and 2.
```

#### Check GPU usage:

```bash
nvidia-smi # `watch nvidia-smi` for live updates
```



## VIM

#### Replace all instances of `foo` with `bar`:

```
:%s/foo/bar/g
```

