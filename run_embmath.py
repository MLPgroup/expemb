from dataclasses import asdict
from simple_parsing import ArgumentParser
from expemb import (
    EmbMathArguments,
    EmbeddingMathematics,
)


def main():
    arg_parser = ArgumentParser("Script to run embedding mathematics.")
    arg_parser.add_arguments(EmbMathArguments, dest = "options")
    args = arg_parser.parse_args()
    args = asdict(args.options)

    embmath = EmbeddingMathematics(**args)
    embmath.run()


if __name__ == "__main__":
    main()
