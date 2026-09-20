import pytest
from self_geometry.ram_limits import scope_command, TOTAL_GIB


def test_each_child_scope_has_real_memory_and_oom_group_limits():
    cmd = scope_command(['python', 'scene_worker.py', '--scene', 'with spaces'])
    assert '--slice=self-geometry.slice' in cmd
    assert '--property=MemoryMax=32G' in cmd
    assert '--property=MemorySwapMax=0' in cmd
    assert cmd[cmd.index('--')+1] == '/usr/bin/python3'
    assert cmd[cmd.index('--')+2].endswith('/scripts/cgroup_exec.py')
    assert cmd[-4:] == ['python', 'scene_worker.py', '--scene', 'with spaces']
    assert TOTAL_GIB == 72


@pytest.mark.parametrize('limit', [0, -1, 73])
def test_invalid_worker_limit_rejected(limit):
    with pytest.raises(ValueError):
        scope_command(['python'], limit)
