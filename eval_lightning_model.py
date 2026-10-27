from eval import Evaluator
from models_list.lightning_log import LightningLogModel


class LightningEvaluator(Evaluator):
    def get_parser(self):
        parser = super().get_parser()        
        parser.add_argument('--repo_path', default=None)
        parser.add_argument('--log', default=None)
        parser.add_argument('--img_size', default=None)
        parser.add_argument('--mode', default='max_size')
        return parser

    def define_model(self):
        return LightningLogModel.get_model(**vars(self.args))

    def define_input_transform(self):
        return LightningLogModel.get_transform(**vars(self.args))
    

def main():
    evaluator = LightningEvaluator()
    evaluator.run()


if __name__ == '__main__':
    main()