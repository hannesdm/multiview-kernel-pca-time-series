import warnings
import os
import logging
import hydra
from omegaconf import DictConfig
from srcs.utils import (
    instantiate,
    instantiate_dict_general,
    str_to_func,
    CustomReporter,
)
from srcs.utils.util import standardize, convert_to_AR
from srcs.model.eval_MV_RKM_nar import Eval_MV_RKM_nar
import ray
from ray import tune
from ray.tune import ExperimentAnalysis

ray.init()
# ray.init(local_mode=True)  # single thread, sequential execution for debugging
assert ray.is_initialized()

warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"
logger = logging.getLogger("parameter_search")


def train_rkm(config, arch=None, data=None, metric=None, ablation_study: bool = False):
    train_data, test_data = data

    # load data
    train_data, train_data_mean, train_data_std = standardize(train_data)
    X_ar, Y_ar = convert_to_AR(
        data=train_data,
        lag=config["lag"],
        n_steps_ahead=config["n_steps_ahead"],
    )

    # train
    rrkm = instantiate(arch, **config)
    rrkm.train(X=X_ar, Y=Y_ar)

    # pred
    n_steps = test_data.shape[0]
    init_vec = convert_to_AR(
        data=train_data[-(config["lag"] + 1) :],
        lag=config["lag"],
        n_steps_ahead=config["n_steps_ahead"],
    )
    Y_pred = rrkm.predict(init_vec=init_vec, n_steps=n_steps, verbose=False)
    Y_pred = (train_data_std * Y_pred) + train_data_mean

    # eval
    if ablation_study:
        # Metrics and plots
        eval_mdl = Eval_MV_RKM_nar(
            config={**config, "metrics": metric},
            model=rrkm,
            test_data=test_data,
            train_data=(train_data_std * train_data) + train_data_mean,
            Y_pred=Y_pred,
        )

        eval_mdl.plot_preds()
        eval_mdl.plot_latents()
    met_result = metric(test_data, Y_pred.squeeze()).item()
    tune.report(**{metric.__name__: met_result})


@hydra.main(config_path="conf/", config_name="parameter_search_nar")
def main(hydra_config: DictConfig) -> None:
    #  Load dataset
    train_data, test_data = instantiate(hydra_config.data)
    train_data = train_data.double()
    test_data = test_data.double()
    metric = str_to_func(hydra_config.hparam_search_metric)

    if "search algorithm" in hydra_config:
        search_algorithm_with_parameters = {
            **hydra_config.search_algorithm,
            "metric": metric.__name__,
        }
        search_algorithm = instantiate(search_algorithm_with_parameters)
    else:
        search_algorithm = None

    try:
        analysis = tune.run(
            tune.with_parameters(
                train_rkm,
                arch=hydra_config.arch,
                data=(train_data, test_data),
                metric=metric,
                ablation_study=hydra_config.ablation_study,
            ),
            metric=metric.__name__,
            mode="min",
            name="exp",
            local_dir=hydra_config["log_dir"],
            search_alg=search_algorithm,
            num_samples=hydra_config["num_experiments"],
            config={**instantiate_dict_general(hydra_config.hyperparameters)},
            progress_reporter=CustomReporter(logger=logger, max_report_frequency=15),
        )
        best_config = analysis.best_config
        best_result = analysis.best_result
        best_log_dir = analysis.best_logdir
    except ray.tune.error.TuneError as e:
        print(e)
        analysis = ExperimentAnalysis(hydra_config["log_dir"])
        best_config = analysis.get_best_config(metric=metric.__name__, mode="min")
        best_result = analysis.get_best_trial(
            metric=metric.__name__, mode="min"
        ).last_result
        best_log_dir = analysis.get_best_logdir(metric=metric.__name__, mode="min")

    if hydra_config.ablation_study:
        pass
    else:
        logger.info("\n\nBest config is: {}".format(best_config))
        logger.info("\n\nBest result: {}".format(best_result))
        logger.info("\n\nBest logdir: {}".format(best_log_dir))

        # load data
        train_data, train_data_mean, train_data_std = standardize(train_data)
        X_ar, Y_ar = convert_to_AR(
            data=train_data,
            lag=best_config["lag"],
            n_steps_ahead=best_config["n_steps_ahead"],
        )

        # train
        rrkm = instantiate(hydra_config.arch, **best_config)
        rrkm.train(X=X_ar, Y=Y_ar)

        # pred
        n_steps = test_data.shape[0]
        init_vec = convert_to_AR(
            data=train_data[-(best_config["lag"] + 1) :],
            lag=best_config["lag"],
            n_steps_ahead=best_config["n_steps_ahead"],
        )
        Y_pred = rrkm.predict(init_vec=init_vec, n_steps=n_steps, verbose=False)
        Y_pred = (train_data_std * Y_pred) + train_data_mean

        # Metrics and plots
        eval_mdl = Eval_MV_RKM_nar(
            config={**best_config, "metrics": hydra_config["metrics"]},
            model=rrkm,
            test_data=test_data,
            train_data=(train_data_std * train_data) + train_data_mean,
            Y_pred=Y_pred,
        )

        eval_mdl.plot_full_data()
        eval_mdl.eval_metrics()
        eval_mdl.plot_preds()
        eval_mdl.plot_latents()


if __name__ == "__main__":
    main()
    ray.shutdown()
