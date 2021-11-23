import time


class Args:
    def __init__(
        self, device, crop_size, model_path, data_path, dataset, mixed, synth_path
    ):
        self.device = device
        self.crop_size = crop_size
        self.model_path = model_path
        self.data_path = data_path
        self.dataset = dataset
        self.mixed = mixed
        self.synth_path = synth_path


if __name__ == "__main__":
    from test import main as test_main

    trained = [
        {
            "type": "gcc",
            "path": "trained/gcc.pth",
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
            print("-" * 10)
            print(f"Testing {tr['path']} on {test['path']}, mixed: {'mixed' in test}")
            start = time.time()
            args = Args(
                0,
                512,
                tr["path"],
                test["path"],
                test["type"],
                "mixed" in test,
                "DATA/processed/SynthAug",
            )
            test_main(args)

            print(f"Time taken: {time.time() - start}")
        print("-" * 10 + f"  Time taken: {time.time() - start_tr}  " + "-" * 10)
        print("-" * 20)
