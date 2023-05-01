import argparse
import json
from pipelines import pipelines_changed
#import sys
#sys.path.append('/home/price/laser/ecg-resnet34/ecg-classification/pipelines')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = json.loads(open(args.config).read())
    pipeline_type = getattr(pipelines_changed, config["type"])

    print("Trainer: ", config["type"], pipeline_type)
    pipeline = pipeline_type(config)
    pipeline.run_pipeline()
