# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Python Built-Ins:
import argparse
import datetime as dt
import json
import os
import glob
import traceback
from types import SimpleNamespace

# External Dependencies:
import awswrangler as wr
import pandas as pd
from sklearn.metrics import accuracy_score


def get_environment():
    """Load configuration variables for SM Model Monitoring job

    See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-contract-inputs.html
    """
    try:
        with open("/opt/ml/config/processingjobconfig.json", "r") as conffile:
            defaults = json.loads(conffile.read())["Environment"]
    except Exception as e:
        traceback.print_exc()
        print(f"Unable to read environment vars from SM processing config file: {e}")
        defaults = {}

    return SimpleNamespace(
        dataset_format=os.environ.get("dataset_format", defaults.get("dataset_format")),
        dataset_source=os.environ.get(
            "dataset_source",
            defaults.get("dataset_source", "/opt/ml/processing/input/endpoint"),
        ),
        end_time=os.environ.get("end_time", defaults.get("end_time")),
        output_path=os.environ.get(
            "output_path",
            defaults.get("output_path", "/opt/ml/processing/output"),
        ),
        publish_cloudwatch_metrics=os.environ.get(
            "publish_cloudwatch_metrics",
            defaults.get("publish_cloudwatch_metrics", "Enabled"),
        ),
        sagemaker_endpoint_name=os.environ.get(
            "sagemaker_endpoint_name",
            defaults.get("sagemaker_endpoint_name"),
        ),
        sagemaker_monitoring_schedule_name=os.environ.get(
            "sagemaker_monitoring_schedule_name",
            defaults.get("sagemaker_monitoring_schedule_name"),
        ),
        start_time=os.environ.get(
            "start_time",
            defaults.get("start_time")),
        baseline_constraints=os.environ.get(
            "baseline_constraints",
            defaults.get("baseline_constraints", "/opt/ml/processing/baseline/constraints/constraints.json")),
        baseline_statistics=os.environ.get(
            "baseline_statistics",
            defaults.get("baseline_statistics", "/opt/ml/processing/baseline/stats/statistics.json")),
        ground_truth_s3_uri_path=os.environ.get(
            "ground_truth_s3_uri_path",
            defaults.get("ground_truth_s3_uri_path"))
    )


def parse_endpoint_capture_output(capture_data_record):
    """Function to parse the endpoint output from the SageMaker Endpoint Capture Data.
    Returns list of label values given 1 or more payloads.

    :param capture_data_record: (dict) Dictionary representing capture data

    """

    output_data_ls = json.loads(capture_data_record["endpointOutput"]["data"])
    label_ls = [inf_record["label"] for inf_record in output_data_ls]

    return label_ls


def parse_metadata_info(event_metadata):
    """Function to parse the event metadata from the SageMaker Endpoint Capture Data.
    Returns inference id

    :param event_metadata: (dict) Dictionary representing event metadata fo endpoint capture

    """

    return event_metadata["inferenceId"]


def flatten_list_of_lists(list_of_lists):
    """Function to flatten a list of lists

    :param list_of_lists: (list) Python list of lists
    """

    return [x for sub_list in list_of_lists for x in sub_list]


def check_for_model_quality_violations(test_constraint_violation=True):
    """Function to read Ground Truth Data, Endpoint Data Capture, and constraints file. Will calculate inference
    data model quality metrics to find any violations. A list of violations are returned.

    :param test_constraint_violation: (bool) True to make accuracy less that constraint violation for testing
    purposes.
    :return:
    """
    list_of_violations = []
    env_vars = get_environment()
    print(f"Starting evaluation with config\n{env_vars}")

    print("Read Ground Truth Data from S3")
    start_date_timestamp = dt.datetime.strptime(env_vars.start_time, "%Y-%m-%dT%H:%M:%SZ")
    timestamp_prefix_parts = [

        start_date_timestamp.year,
        start_date_timestamp.month,
        start_date_timestamp.day,
        start_date_timestamp.hour
    ]
    data_partition_prefix = "/".join([str(ts) for ts in timestamp_prefix_parts])

    ground_truth_path = f"{env_vars.ground_truth_s3_uri_path}/{data_partition_prefix}"
    ground_truth_data = wr.s3.read_json(ground_truth_path)

    print("Read SageMaker Endpoint Data Capture")

    json_data_capture_files = glob.glob(
        f"{env_vars.dataset_source}/{env_vars.sagemaker_endpoint_name}/*/*/*/*/*/*.jsonl",
        recursive=True
    )

    json_data_ls = [pd.read_json(filename, lines=True) for filename in json_data_capture_files]
    df = pd.concat(json_data_ls, axis=0)

    print("Read Constraints File")
    with open(env_vars.baseline_constraints, 'r') as f:
        constraints_data = json.load(f)

    print("Merge Ground Truth with endpoint data capture")
    # get labels from each SM Endpoint Payload
    label_data = list(df["captureData"].apply(parse_endpoint_capture_output))

    # get inference Ids from each SM endpoint payload
    inference_ids = list(df["eventMetadata"].apply(parse_metadata_info))
    inference_ids_ls = [[inf_id] * len(label_ls) for inf_id, label_ls in zip(inference_ids, label_data)]

    # Create payload index ls give each SM endpoint payload
    payload_index_ls = [list(range(len(label_ls))) for label_ls in label_data]

    # create inference dataframe
    inference_df = pd.DataFrame({
        "InferenceId": flatten_list_of_lists(inference_ids_ls),
        "payload_index": flatten_list_of_lists(payload_index_ls),
        "prediction_label": flatten_list_of_lists(label_data)
    })
    # convert Inference Id to int for downstream join
    inference_df["InferenceId"] = inference_df["InferenceId"].astype(int)

    # merge ground truth with inference data
    eval_df = pd.merge(ground_truth_data, inference_df, on=["InferenceId", "payload_index"], how="inner")

    # calculate accuracy score
    inf_data_accuracy_score = accuracy_score(eval_df["groundTruthLabel"], eval_df["prediction_label"])
    print(f"Data capture accuracy inference score: {inf_data_accuracy_score}")

    if test_constraint_violation:
        inf_data_accuracy_score = -1

    accuracy_threshold = constraints_data["accuracy"]["threshold"]
    if inf_data_accuracy_score < accuracy_threshold:
        list_of_violations.append(
            {
                "feature_name": "forest-coverage-inf-accuracy",
                "constraint_check_type": "mqm_drift_check",
                "value": inf_data_accuracy_score,
                "description": f"Model Quality Alert - accuracy_score: {inf_data_accuracy_score} "
                               f"is less than validation_accuracy: {accuracy_threshold}"
            }
        )

    return list_of_violations


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-violation-tests", action='store_true')
    args, _ = parser.parse_known_args()

    env = get_environment()

    violations = check_for_model_quality_violations(test_constraint_violation=args.create_violation_tests)

    print("Checking for constraint violations...")
    print(f"Violations: {violations if len(violations) else 'None'}")

    print("Writing violations file...")
    with open(os.path.join(env.output_path, "constraint_violations.json"), "w") as outfile:
        outfile.write(json.dumps(
            {"violations": violations},
            indent=4,
        ))

    print("Writing overall status output...")
    with open("/opt/ml/output/message", "w") as outfile:
        if len(violations):
            msg = ''
            for v in violations:
                msg += f"CompletedWithViolations: {v['description']}"
                msg += "\n"
        else:
            msg = "Completed: Job completed successfully with no violations."
        outfile.write(msg)
        print(msg)

    if env.publish_cloudwatch_metrics == "Enabled":
        print("Writing CloudWatch metrics...")
        # format violations information for CW metrics
        cw_metric_ls = []
        for v in violations:
            cw_metric_ls.append(
                {
                    "MetricName": v["feature_name"],  # Required
                    "Timestamp": dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%dT%H:%M:%SZ"),  # Required
                    "Dimensions": [{"Name": "MonitoringSchedule", "Value": env.sagemaker_monitoring_schedule_name}],
                    "Value": v["value"]
                }
            )
        with open("/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl", "a+") as outfile:
            # One metric per line (JSONLines list of dictionaries)
            # Remember these metrics are aggregated in graphs, so we report them as statistics on our dataset
            for cw_metric in cw_metric_ls:
                cw_line = f"{json.dumps(cw_metric, indent=4)}\n"
                outfile.write(cw_line)
    print("MQM Job complete!")
