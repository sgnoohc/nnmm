# nnmm

    srun --partition=gpu --gpus=1 --constraint=a100 --cpus-per-gpu=1 --account=avery --qos=avery --pty bash -i

    source setup.sh
    make
    ./run_test

    python validate.py # for comparing results
