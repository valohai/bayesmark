import logging

import bayesmark.cmd_parse as cmd
from bayesmark.cmd_parse import CmdArgs
from bayesmark.constants import ARG_DELIM
from bayesmark.serialize import XRSerializer
from bayesmark.util import str_join_safe

logger = logging.getLogger(__name__)


def main():  # pragma: main
    description = "Train surrogate models for bayesmark"
    # TODO choose args
    args = cmd.parse_args(cmd.experiment_parser(description))

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    test_case = str_join_safe(ARG_DELIM, (args[CmdArgs.classifier], args[CmdArgs.data], args[CmdArgs.metric]))

    # Load in the eval data and sanity check
    suggest_ds, meta = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=test_case)
    logger.info("Meta data from source file: %s" % str(meta["args"]))

    # TODO cross check meta data


if __name__ == "__main__":
    main()  # pragma: main
