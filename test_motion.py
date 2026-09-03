from pathlib import Path
import numpy as np
from motion import estimate

ROOT = Path(__file__).resolve().parent


def load(i):
    return np.fromfile(ROOT / 'cache' / 'raw' / f'00_{i:06d}.bin',
                       np.float32).reshape(-1, 4)


def main():
    for i in range(7):
        r = estimate(load(i), load(i + 1), dt=0.1)
        print(f'{i:06d}->{i+1:06d}  '
              f'T=({r.tx:+.2f},{r.ty:+.2f}) m  '
              f'speed={r.speed_mps:.2f} m/s ({r.speed_kmh:.1f} km/h)  '
              f'conf={r.confidence:.2f}')


if __name__ == '__main__':
    main()
