"""Test on Parallel computation."""

from joblib import Parallel, delayed
from time import sleep
from tqdm import tqdm

if __name__ == "__main__":
    # r = Parallel(n_jobs=3)(delayed(sleep)(0.1) for _ in tqdm(range(100)))
    shared_set = set()

    def collect(x, s):
        sleep(0.1)
        shared_set.add(x * s)

    Parallel(n_jobs=3, require="sharedmem")(
        delayed(collect)(i, s=3) for i in tqdm(range(100))
    )

    print(shared_set)
