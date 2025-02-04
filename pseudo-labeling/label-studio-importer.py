#!/usr/bin/env python3

import json
import argparse
import requests

def upload_tasks_to_label_studio(tasks_json, label_studio_url, label_studio_api_key, project_id):
    """
    Uploads tasks in Label Studio JSON format to a Label Studio project via REST API.
    """
    url = f"{label_studio_url}/api/projects/{project_id}/import"
    headers = {
        "Authorization": f"Token {label_studio_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=tasks_json)
    if response.status_code in [200, 201]:
        print(f"Successfully uploaded {len(tasks_json)} tasks to Label Studio (project {project_id}).")
    else:
        print(f"Failed to upload tasks. Status: {response.status_code}\nReason: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Upload tasks JSON to Label Studio project.")
    parser.add_argument("--tasks-json", type=str, required=True,
                        help="Path to the JSON file with Label Studio tasks.")
    parser.add_argument("--label-studio-url", type=str, default="http://localhost:8080",
                        help="Base URL of the Label Studio instance.")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Your Label Studio User or Project API Key.")
    parser.add_argument("--project-id", type=int, required=True,
                        help="The target Label Studio project ID where tasks will be imported.")
    args = parser.parse_args()

    # 1) Load tasks from JSON
    with open(args.tasks_json, "r") as f:
        tasks = json.load(f)

    # 2) Upload tasks
    upload_tasks_to_label_studio(
        tasks_json=tasks,
        label_studio_url=args.label_studio_url,
        label_studio_api_key=args.api_key,
        project_id=args.project_id
    )

if __name__ == "__main__":
    main()
