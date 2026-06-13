from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
# Optional environments that require external packages (VirtualTaobao/GAN_SD,
# RecoGym, RecSim, RL4RS). Exclude them from `from env import *` so the core
# KuaiRand environments import cleanly without those deps installed; import the
# specific module explicitly if you have the dependencies.
_optional_prefixes = ('VirTB', 'Recogym', 'Recsim', 'RL4RS')
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
    and not basename(f).startswith(_optional_prefixes)
]
