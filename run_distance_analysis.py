from dataclasses import asdict
from simple_parsing import ArgumentParser
from expemb import (
    DistanceAnalysisArguments,
    DistanceAnalysis,
)


def main():
    arg_parser = ArgumentParser("Script to run embedding mathematics.")
    arg_parser.add_arguments(DistanceAnalysisArguments, dest = "options")
    args = arg_parser.parse_args()
    args = asdict(args.options)

    embmath = DistanceAnalysis(**args)
    embmath.run()


if __name__ == "__main__":
    main()
