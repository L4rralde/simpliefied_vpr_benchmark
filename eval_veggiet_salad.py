from eval import Evaluator
from models_list.veggiet_salad import VeggietModel


class VeggietSaladEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.args.infer_batch_size = 1

    def get_parser(self):
        parser = super().get_parser()
        parser.add_argument('--repo_path', default=None)
        parser.add_argument('--model', default=None)
        parser.add_argument('--ckpt', default=None)
        return parser

    def define_model(self):
        return VeggietModel.get_model(**vars(self.args))

    def define_input_transform(self):
        return VeggietModel.get_transform(**vars(self.args))


def main():
    evaluator = VeggietSaladEvaluator()
    evaluator.run()


if __name__ == '__main__':
    main()
