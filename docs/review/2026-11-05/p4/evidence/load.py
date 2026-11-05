import multiprocessing as m, time, sys
def burn(t):
    end=time.time()+t
    while time.time()<end: sum(i*i for i in range(10000))
if __name__=='__main__':
    n=int(sys.argv[1]); t=float(sys.argv[2])
    ps=[m.Process(target=burn,args=(t,)) for _ in range(n)]
    [p.start() for p in ps]; [p.join() for p in ps]
