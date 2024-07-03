import os
from enum import Enum

import luigi

from deepclean.couplings import subtraction_problems

problems = {k: k for k in subtraction_problems}
ifos = {i: i for j in subtraction_problems.values() for i in j}
ifos = Enum("ifo", ifos)
problems = Enum("problem", problems)


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", "ml4gw"))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "deepclean"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))


class deepclean(luigi.Config):
    ifo = luigi.EnumParameter(enum=ifos)
    problem = luigi.EnumListParameter(enum=problems)
    strain_channel = luigi.Parameter(default="GDS-CALIB_STRAIN")
    container_root = luigi.Parameter(
        default=os.getenv("DEEPCLEAN_CONTAINER_ROOT", "")
    )

    wandb = wandb()

    @property
    def couplings(self):
        ifo = self.ifo.value
        return [subtraction_problems[i.value][ifo] for i in self.problem]

    @property
    def channels(self):
        return [j for i in self.couplings for j in i.channels]

    @property
    def freq_low(self):
        return [i.freq_low for i in self.couplings]

    @property
    def freq_high(self):
        return [i.freq_high for i in self.couplings]
