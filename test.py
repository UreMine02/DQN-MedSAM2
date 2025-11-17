import tracemalloc
from collections import deque

def main():
    q = deque(maxlen=1000)
    
    for i in range(1000000):
        q.append(i)
    
if __name__ == '__main__':
    tracemalloc.start()
    main()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')

    print("[ Top 20 ]")
    for stat in top_stats[:20]:
        print(stat)