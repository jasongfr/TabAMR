import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')
import logging
import tqdm
from models import build_model
from processors import build_processor
from utils import set_seed
from runner.runner import Runner
import torch

torch.cuda.set_device(0)
print(torch.cuda.device_count())
print(torch.cuda.is_available())


logger = logging.getLogger(__name__)


def run(args, model, processor, optimizer, scheduler):
    set_seed(args)
    print("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')
    print("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    runner = Runner(
        cfg=args,
        data_samples=[dev_examples, test_examples],
        data_features=[dev_features, test_features],
        data_loaders=[dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
        processor=processor,
    )
    runner.run()


def main():
    from config_parser import get_args_parser
    args = get_args_parser()

    if not args.inference_only:
        print(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

            logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt='%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    else:
        logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    set_seed(args)

    model, tokenizer, optimizer, scheduler = build_model(args)
    model.cuda(device=0)

    processor = build_processor(args, tokenizer)

    logger.info("Training/evaluation parameters %s", args)
    run(args, model, processor, optimizer, scheduler)


if __name__ == "__main__":
    main()