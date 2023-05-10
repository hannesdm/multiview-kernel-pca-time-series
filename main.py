import logging
import os
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from srcs.model.eval_MV_RKM_nar import Eval_MV_RKM_nar
from srcs.utils.util import convert_to_AR, standardize, instantiate

warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"
logger = logging.getLogger("main")


@hydra.main(config_path="conf/", config_name="train_rrkm_nar")
def main(config: DictConfig) -> None:
    # Train model ===================================
    params = OmegaConf.to_container(config.hyperparameters)

    logger.info("\n\n Config: {}\n".format(config))

    # Load & standardize data
    train_data, test_data = instantiate(config.data)
    train_data, train_data_mean, train_data_std = standardize(train_data.double())
    X_ar, Y_ar = convert_to_AR(
        data=train_data, lag=params["lag"], n_steps_ahead=params["n_steps_ahead"]
    )

    # Train
    rrkm = instantiate(config.arch, **params)
    rrkm.train(X=X_ar, Y=Y_ar)

    # Test
    n_steps = test_data.shape[0]
    init_vec = convert_to_AR(
        data=train_data[-(params["lag"] + 1):],
        lag=params["lag"],
        n_steps_ahead=params["n_steps_ahead"],
    )
    Y_pred = rrkm.predict(init_vec=init_vec, n_steps=n_steps, mode="x_to_y")
    Y_pred = (train_data_std * Y_pred) + train_data_mean

    # Metrics and plots
    eval_mdl = Eval_MV_RKM_nar(
        config=config,
        model=rrkm,
        test_data=test_data,
        train_data=(train_data_std * train_data) + train_data_mean,
        Y_pred=Y_pred,
    )

    eval_mdl.plot_full_data()
    eval_mdl.eval_metrics()
    eval_mdl.plot_preds()
    eval_mdl.plot_latents()
    eval_mdl.plot_latent_components()

    logger.info("--END--")


if __name__ == "__main__":
    main()
