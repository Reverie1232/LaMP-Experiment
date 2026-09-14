from evaluation import LaMPEvaluation
import argparse
import json
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--golds_json",
    default="./data/news_headline/test_100/user_test_100_label.json",
    help="Address to all gold labels for the task as a json file"
)

parser.add_argument(
    "--preds_json", 
    default="./output/news_headline/all-interaction-preference-lora-merge/k1-profile-top3.json", 
    help="Address to all predictions for the task as a json file"
)
parser.add_argument("--task_name", help="[LaMP_1, LaMP_2, LaMP_3, LaMP_4, LaMP_5, LaMP_6, LaMP_7]")
parser.add_argument("--output_file", help="Address to the results file")

if __name__ == "__main__":

    opts = parser.parse_args()
    
    evaluator = LaMPEvaluation(single_gold_json_file_addr=opts.golds_json)
    results = evaluator.evaluate_task(opts.preds_json, opts.task_name)
    print(results)
    # with open(opts.output_file, "w") as file:
    #     json.dump(results, file)
