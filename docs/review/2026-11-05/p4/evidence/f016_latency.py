"""Round-trip latency of ColdSupervisor.__call__ with the test's own fixture worker (call_seconds large to measure)."""
import sys, time, tempfile, json
sys.path.insert(0, '.')
from tests.test_study_stimulus_supervisor import fixture_worker
from scripts.study_operator.stimulus_supervisor import ColdSupervisor
def main():
    root = tempfile.mkdtemp()
    lat = []
    with ColdSupervisor(dict(root=root), admission=lambda r: None, poll=lambda: None, worker_target=fixture_worker,
                        preparation_seconds=30, call_seconds=30) as runner:
        for i in range(int(sys.argv[1])):
            t = time.perf_counter(); runner(i); lat.append(time.perf_counter() - t)
    lat.sort()
    print(json.dumps({'n': len(lat), 'p50_ms': round(lat[len(lat)//2]*1000, 2), 'p99_ms': round(lat[int(len(lat)*.99)]*1000, 2),
                      'max_ms': round(lat[-1]*1000, 2), 'over_100ms': sum(x > .1 for x in lat)}))

if __name__ == '__main__':
    main()
