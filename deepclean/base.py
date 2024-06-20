import os
from collections.abc import Callable
from pathlib import Path

import law
import luigi
from law.contrib import singularity

from deepclean.config import deepclean as Config
from deepclean.utils import stream_command

root = Path(__file__).resolve().parent.parent


class DeepCleanSandbox(singularity.SingularitySandbox):
    sandbox_type = "deepclean"

    @property
    def data_directories(self):
        return ["/cvmfs", "/hdfs", "/gpfs", "/ceph", "/hadoop", "/archive"]

    def _get_volumes(self):
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[str(root)] = "/opt/deepclean"

        # bind data directories if they exist on this cluster
        for dir in self.data_directories:
            if os.path.exists(dir):
                volumes[dir] = dir

        return volumes

class DeepCleanTask(law.SandboxTask):
    image = luigi.Parameter()
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    cfg = Config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.isabs(self.image):
            self.image = os.path.join(Config().container_root, self.image)

        if not os.path.exists(self.image):
            raise ValueError(
                f"Could not find path to container image {self.image}"
            )

    @property
    def ifo(self):
        return self.cfg.ifo.value

    @property
    def strain_channel(self):
        return f"{self.ifo}:{self.cfg.strain_channel}"

    @property
    def witnesses(self):
        return self.cfg.channels

    @property
    def sandbox(self):
        return f"deepclean::{self.image}"

    @property
    def singularity_args(self) -> Callable:
        def arg_getter():
            if self.gpus:
                return ["--nv"]
            return []

        return arg_getter

    def sandbox_env(self, _):
        env = {}
        for envvar, value in os.environ.items():
            if envvar.startswith("DEEPCLEAN_"):
                env[envvar] = value

        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
        return env

    @property
    def python(self) -> str:
        return "/usr/local/bin/python"

    @property
    def command(self) -> str:
        return [self.python, "-c", "print('Hello world')"]

    def run(self):
        stream_command(self.command)
