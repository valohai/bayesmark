import json

from bayesmark.cmd_parse import CmdArgs, general_parser, parse_args
from bayesmark.constants import ITER, MEAN_SCORE, METHOD, NORMED_MEAN
from bayesmark.serialize import XRSerializer

METHOD_TO_LOG = "BlackBoxOptimizer"


def main():
    description = "Output final score for tools such as Valohai."
    args = parse_args(general_parser(description))

    # Load in the eval data and sanity check
    summary, meta = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=MEAN_SCORE)

    final_score = summary[NORMED_MEAN][{ITER: -1, METHOD: METHOD_TO_LOG}]

    print("final score:")
    print()
    print(json.dumps({"final_normalized_mean": final_score}))


if __name__ == "__main__":
    main()  # pragma: main
