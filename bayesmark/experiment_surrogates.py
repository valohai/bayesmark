import logging
import os.path
import pickle as pkl
from itertools import product

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

import bayesmark.cmd_parse as cmd
import bayesmark.xr_util as xru
from bayesmark.cmd_parse import CmdArgs
from bayesmark.constants import (
    ARG_DELIM,
    DATA_LOADER_NAMES,
    EVAL_RESULTS,
    ITER,
    METHOD,
    METRICS,
    MODEL_NAMES,
    SUGGEST,
    TEST_CASE,
    TRIAL,
)
from bayesmark.data import METRICS_LOOKUP, ProblemType, get_problem_type
from bayesmark.serialize import XRSerializer
from bayesmark.sklearn_funcs import MODELS_CLF, MODELS_REG
from bayesmark.space import JointSpace
from bayesmark.util import str_join_safe, strict_sorted

logger = logging.getLogger(__name__)


def main():  # pragma: main
    description = "Train surrogate models for bayesmark"
    args = cmd.parse_args(cmd.launcher_parser(description))

    # Setup logging
    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    # Setup list of models, data, metrics to iterate over
    c_list = strict_sorted(MODEL_NAMES if args[CmdArgs.classifier] is None else args[CmdArgs.classifier])
    d_list = strict_sorted(DATA_LOADER_NAMES if args[CmdArgs.data] is None else args[CmdArgs.data])
    m_set = set(METRICS if args[CmdArgs.metric] is None else args[CmdArgs.metric])
    m_lookup = {problem_type: sorted(m_set.intersection(mm)) for problem_type, mm in METRICS_LOOKUP.items()}
    assert all(
        (len(m_lookup[get_problem_type(data)]) > 0) for data in d_list
    ), "At one metric needed for each problem type of data sets"

    # Load the function evaluations
    all_perf, meta = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=EVAL_RESULTS)
    logger.info("Meta data from source file: %s" % str(meta["args"]))
    all_perf = xru.only_dataarray(all_perf)
    assert isinstance(all_perf, xr.DataArray)
    assert all_perf.dims == (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)
    assert xru.is_simple_coords(all_perf.coords, dims=(ITER, SUGGEST, TRIAL))
    assert not np.any(np.isnan(all_perf.values))

    for model, data in product(c_list, d_list):
        problem_type = get_problem_type(data)
        assert problem_type in (ProblemType.clf, ProblemType.reg)
        for metric in m_lookup[problem_type]:
            # Build name of test case
            test_case = str_join_safe(ARG_DELIM, (model, data, metric))

            # Load in the suggestions
            suggest_ds, meta_t = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=test_case)
            logger.info("Meta data from source file: %s" % str(meta["args"]))

            # Validate/cross check meta data
            assert meta == meta_t, "meta data should between suggest and eval files"

            # TODO pull out all non-IO parts to routine

            # Select out data for this problem
            perf_curr = all_perf.sel({TEST_CASE: test_case})
            suggest_ds = suggest_ds.sel({TEST_CASE: test_case})

            # Setup space warping
            _, _, api_config = MODELS_CLF[model] if problem_type == ProblemType.clf else MODELS_REG[model]
            space = JointSpace(api_config)

            # Get out numeric np data from xr logs
            suggest_data = {}
            y = perf_curr.values
            for kk in suggest_ds:
                suggest_da = suggest_ds[kk]
                assert isinstance(suggest_da, xr.DataArray)
                assert suggest_da.dims == (ITER, SUGGEST, METHOD, TRIAL)
                assert xru.is_simple_coords(suggest_da.coords, dims=(ITER, SUGGEST, TRIAL))
                assert not np.any(np.isnan(suggest_da.values))
                assert xru.coord_compat((perf_curr, suggest_da), (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL))

                X = suggest_ds[kk].values
                assert X.shape == y.shape
                suggest_data[kk] = X.ravel(order="C")
            y = y.ravel(order="C")
            X = [{param: suggest_data[param][ii].item() for param in space.param_list} for ii in range(len(y))]
            X = space.warp(X)  # Use space class to get cartesian

            # More validation on x,y
            assert X.ndim == 2
            assert y.shape == (X.shape[0],)
            assert not np.any(np.isnan(X))
            assert not np.any(np.isnan(y))

            # Train an surrogate model
            # TODO apply BO on the hypers!
            # Will also need to consider standardization if we move away from RF
            surr_model = RandomForestRegressor(n_estimators=100)
            surr_model.fit(X, y)

            # Save the surrogate model for this test case
            path = os.path.join(args[CmdArgs.data_root], test_case + ".pkl")
            with open(path, "wb") as f:
                pkl.dump(surr_model, f)


if __name__ == "__main__":
    main()  # pragma: main
