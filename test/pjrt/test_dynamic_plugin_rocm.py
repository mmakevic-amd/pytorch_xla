import concurrent.futures

from absl.testing import absltest
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.experimental import plugins
import torch_xla.runtime as xr
import torch_xla_rocm_plugin

plugins.register_plugin('ROCM', torch_xla_rocm_plugin.RocmPlugin())
plugins.use_dynamic_plugins()


class TestDynamicRocmPlugin(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    xr.set_device_type('ROCM')

  @staticmethod
  def _assert_gpus_exist(index=0):
    del index
    assert len(xm.get_xla_supported_devices('ROCM')) > 0

  def test_single_process(self):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
      executor.submit(self._assert_gpus_exist).result()

  def test_spawn(self):
    xmp.spawn(self._assert_gpus_exist)


if __name__ == '__main__':
  absltest.main()
