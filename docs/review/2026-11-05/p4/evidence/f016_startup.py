import sys, time, tempfile, json
sys.path.insert(0, '.')
def main():
    from tests.test_study_stimulus_supervisor import fixture_worker
    from scripts.study_operator.stimulus_supervisor import ColdSupervisor
    ts = []
    for _ in range(int(sys.argv[1])):
        s = ColdSupervisor(dict(root=tempfile.mkdtemp()), admission=lambda r: None, poll=lambda: None,
                           worker_target=fixture_worker, preparation_seconds=120, call_seconds=30)
        t = time.perf_counter(); s.__enter__(); ts.append(time.perf_counter() - t); s.__exit__(None, None, None)
    ts.sort(); print(json.dumps({'n': len(ts), 'p50_s': round(ts[len(ts)//2], 2), 'max_s': round(ts[-1], 2), 'over_3s': sum(x > 3 for x in ts)}))
if __name__ == '__main__':
    main()
