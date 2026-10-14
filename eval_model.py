import inspect
import logging

from eval import Evaluator
import models


class GeneralEvaluator(Evaluator):
    def get_parser(self):
        parser = self.get_parser()
        parser.add_argument('--model', required=True)
        parser.add_argument('--ckpt', default=None)
        parser.add_argument('--repo_path', default=None)

        return parser

    def get_model_namespace(self) -> models.Model:
        model_names = [obj.__name__.upper() for obj in models.ALL_MODELS]
        try:
            model_idx = model_names.index(self.args.model.upper())
        except ValueError as E:
            logging.error(f"{self.args.model} is not a valid model name. See {model_names}")
            raise E

        model_ns: models.Model = models.ALL_MODELS[model_idx]
        return model_ns

    def define_model(self):
        model_ns: models.Model = self.get_model_namespace()
        return (
            model_ns.get_model(**self.args),
            model_ns.__name__,
            model_ns.DESC_DIM
        )

    def define_input_transform(self):
        model_ns: models.Model = self.get_model_namespace()
        return model_ns.get_transform(**self.args)


def main():
    evaluator = GeneralEvaluator()
    evaluator.run()


if __name__ == '__main__':
    main()