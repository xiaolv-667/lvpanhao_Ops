"""The original standalone TileOPs msprof-op collection contract."""

from tileops.toolchains.cann_profile import benchmark_profile_result


class MsprofOp:
    profiler = "msprof"
    benchmark_profile_result = staticmethod(benchmark_profile_result)

    def benchmark_profiler_args(self, kernel, output, launch_count, warm_up):
        return [
            self.profiler,
            "op",
            *([f"--kernel-name={kernel}"] if kernel is not None else []),
            f"--output={output}",
            f"--launch-count={launch_count}",
            f"--warm-up={warm_up}",
            "--dump=off",
            "--aic-metrics=Roofline",
        ]

    def execution_command(self, command, *, output_dir=None):
        return list(command)
