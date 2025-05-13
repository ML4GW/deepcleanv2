#!/home/chiajui.chou/.cache/pypoetry/virtualenvs/train-Mjstll9c-py3.10/bin/python
import os
import subprocess
import sys
from queue import Queue
from threading import Thread

from deepclean.couplings import subtraction_problems

# Environment variables
HOME = "/home/chiajui.chou"
DEEPCLEAN_CONTAINER_ROOT = f"{HOME}/images/deepclean"
DATA_DIR = f"{HOME}/deepclean/data/CDC_test-120Hz"
RESULTS_DIR = f"{HOME}/deepclean/results"
DEEPCLEAN_IFO = "H1"
DEEPCLEAN_PROBLEM = "120Hz"
GPU_INDEX = 1
# Original deepclean.tasks.Train
# luigi.cfg
cfg = dict.fromkeys(['deepclean', 'core'])
for key in cfg:
    cfg[key] = dict()
cfg['deepclean']['ifo'] = DEEPCLEAN_IFO
cfg['deepclean']['problem'] = DEEPCLEAN_PROBLEM
cfg['deepclean']['strain_channel'] = "GDS-CALIB_STRAIN"
cfg['core']['local_scheduler'] = True
cfg['core']['module'] = "deepclean"
# deepclean.config.deepclean
ifo = cfg['deepclean']['ifo']
problem = [cfg['deepclean']['problem']]
strain_channel = f"{ifo}:{cfg['deepclean']['strain_channel']}"
container_root = DEEPCLEAN_CONTAINER_ROOT
# subtraction problems
couplings = [subtraction_problems[i][ifo] for i in problem]
witnesses = [j for i in couplings for j in i.channels]
freq_low = [i.freq_low for i in couplings]
freq_high = [i.freq_high for i in couplings]
# train.sh and Train task
image = "train.sif"
gpus = GPU_INDEX
data_fname = f"{DATA_DIR}/deepclean-1378402219-3072.hdf5"
train_config = f"{HOME}/deepcleanv2/projects/train/config.yaml"
output_dir = f"{RESULTS_DIR}/O4-CDC_120Hz_outlaw"
config = train_config
channels = [strain_channel] + witnesses
freq_low = freq_low
freq_high = freq_high

def read_stream(stream, process, q):
    stream = getattr(process, stream)
    try:
        it = iter(stream.readline, b"")
        while True:
            try:
                line = next(it)
            except StopIteration:
                break
            q.put(line.decode())
    finally:
        q.put(None)

def stream_process(process):
    q = Queue()
    args = (process, q)
    streams = ["stdout", "stderr"]
    threads = [Thread(target=read_stream, args=(i,) + args) for i in streams]
    for t in threads:
        t.start()

    for _ in range(2):
        for line in iter(q.get, None):
            sys.stdout.write(line)

def stream_command(command: list[str]):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=os.environ
    )
    stream_process(process)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
command = [
    "python",
    "-m",
    "train",
    "--config",
    train_config,
    "--data.fname",
    data_fname,
    "--data.channels",
    "[" + ",".join(channels) + "]",
    "--data.freq_low",
    str(freq_low),
    "--data.freq_high",
    str(freq_high),
]
command.append(f"--trainer.logger.save_dir={output_dir}")
print(command)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

stream_command(command)