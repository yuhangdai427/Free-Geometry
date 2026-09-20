import json
import multiprocessing as mp
import time

from self_geometry.gpu_queue import reserve_gpu, reservation, CAPACITY_GIB


def worker(path, size, messages, release):
    with reserve_gpu(size, path):
        messages.put(size)
        release.wait(15)


def test_gpu_admission_overlap_and_exclusive_wait(tmp_path):
    ctx = mp.get_context('fork')
    messages = ctx.Queue()
    release = ctx.Event()
    path = tmp_path/'admission.json'
    procs = [ctx.Process(target=worker, args=(path, 42, messages, release)) for _ in range(2)]
    exclusive = ctx.Process(target=worker, args=(path, CAPACITY_GIB, messages, release))
    try:
        for p in procs:
            p.start()
        assert messages.get(timeout=10) == 42
        assert messages.get(timeout=10) == 42
        exclusive.start()
        deadline = time.monotonic() + 10
        while str(exclusive.pid) not in json.loads(path.read_text()) and time.monotonic() < deadline:
            time.sleep(.05)
        state = json.loads(path.read_text())
        assert state[str(exclusive.pid)]['status'] == 'waiting'
        release.set()
        assert messages.get(timeout=10) == CAPACITY_GIB
        for p in procs + [exclusive]:
            p.join(10)
            assert p.exitcode == 0
        assert json.loads(path.read_text()) == {}
    finally:
        release.set()
        for p in procs + [exclusive]:
            if p.pid and p.is_alive():
                p.terminate()
                p.join()


def test_unknown_workloads_exclusive_and_square_views_reserve_more():
    c = dict(method='test3r', model='da3', image_size=504)
    assert reservation(c) == CAPACITY_GIB
    assert reservation(dict(c, method='tco'), {'shape': [100,3,378,504]}) == CAPACITY_GIB
    assert reservation(c, {'shape': [100,3,378,504]}) == 42
    assert reservation(c, {'shape': [23,3,504,504]}) == 56
