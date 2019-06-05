# Miscellaneous Details

## SSH

Creating an SSH key:

```bash
ssh-keygen -t rsa -b 4096 -C "[email]"
```

Logging in via SSH:

```bash
ssh <user@remote>
```



## GPU Usage

Specify which GPU to train on:

```bash
export CUDA_VISIBLE_DEVICES=1,2 # This selects GPUs 1 and 2.
```

Check GPU usage:

```bash
nvidia-smi # `watch nvidia-smi` for live updates
```



## Tensorboard

View tensorboard results on localhost:

1. From the local machine:

   ```bash
   ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
   ```

2. From the remote machine:

   ```bash
   tensorboard --logdir <path> --port 6006
   ```

3. View locally at `localhost:16006`

