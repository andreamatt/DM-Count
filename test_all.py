import os
import time

if __name__ == "__main__":

    trained = [
        {
            "type": "gcc",
            "path": "gcc.pth",
        },
        {
            "type": "synth",
            "path": "trained/synth.pth",
        },
        {
            "type": "synth",
            "path": "trained/synthaug.pth",
        },
        {
            "type": "nwpu",
            "path": "trained/nwpu_normal.pth",
        },
        {
            "type": "nwpu",
            "path": "trained/nwpu_mixed.pth",
        },
        {
            "type": "nwpu",
            "path": "trained/nwpu_noval.pth",
        },
        {
            "type": "qnrf",
            "path": "trained/qnrf_normal.pth",
        },
        {
            "type": "qnrf",
            "path": "trained/qnrf_mixed.pth",
        },
        {
            "type": "qnrf",
            "path": "trained/qnrf_noval.pth",
        },
    ]

    test_types = [
        {
            "type": "gcc",
            "path": "DATA/processed/GCC",
        },
        {
            "type": "synth",
            "path": "DATA/processed/Synth",
        },
        {
            "type": "synth",
            "path": "DATA/processed/SynthAug",
        },
        {
            "type": "nwpu",
            "path": "DATA/processed/NWPU",
        },
        {
            "type": "nwpu",
            "path": "DATA/processed/NWPU",
            "mixed": True,
        },
        {
            "type": "qnrf",
            "path": "DATA/processed/QNRF",
        },
        {
            "type": "qnrf",
            "path": "DATA/processed/QNRF",
            "mixed": True,
        },
    ]

    for tr in trained:
        print(f"Testing {tr['path']}")
        start_tr = time.time()
        for test in test_types:
            start = time.time()
            print("-" * 10)
            print(f"On {test['path']}, mixed: {'mixed' in test}")
            if "mixed" in test:
                os.system(
                    f"python ./test.py --model-path {tr['path']} --data-path {test['path']} --dataset {test['type']} --mixed true --synth-path DATA/processed/SynthAug"
                )
            else:
                os.system(
                    f"python ./test.py --model-path {tr['path']} --data-path {test['path']} --dataset {test['type']}"
                )
            print(f"Time taken: {time.time() - start}")
        print("-" * 10 + f"  Time taken: {time.time() - start_tr}  " + "-" * 10)
        print("-" * 20)
