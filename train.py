from pytorch_lightning import Trainer
from utils.build_model import build_model
from utils.build_data import build_data
from utils.defaults import argument_parser, load_config


def main():
    args = argument_parser().parse_args()
    configs = load_config(args.cfg)

    model = build_model(configs['model'])
    model = model(configs)

    data = build_data(configs['datamodule'])
    data = data(configs['dataset'])

    trainer = Trainer(
        # gpus=1,
        max_epochs=10,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
