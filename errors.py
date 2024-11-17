"""Module for parsing pre-commit output and saving issues to a JSON file."""

import json
import re
from typing import Any, Dict, List


def parse_pre_commit_output(output: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse the output of pre-commit and return a structured list of issues."""
    issues: List[Dict[str, Any]] = []
    current_hook: Dict[str, Any] = None

    for line in output.splitlines():
        hook_match = re.match(r"^([\w\s]+)\.+\(([\w\s]+)\)(Skipped|Failed)?", line)
        if hook_match:
            if current_hook:
                issues.append(current_hook)
            hook_name, status, additional_status = hook_match.groups()
            status = (
                status.strip()
                if additional_status is None
                else f"{status} {additional_status}"
            )
            current_hook = {
                "hook": hook_name.strip(),
                "status": status.strip(),
                "details": [],
            }
            continue

        file_issue_match = re.match(r"^-? (Fixing|reformatted) (.+)", line)
        if file_issue_match and current_hook:
            action, file_path = file_issue_match.groups()
            current_hook["details"].append(
                {"file": file_path, "issue": action.capitalize()}
            )
            continue

        flake8_match = re.match(r"^(.+):(\d+):\d+: (\w+) (.+)", line)
        if flake8_match:
            if current_hook is None or "flake8" not in current_hook["hook"]:
                current_hook = {"hook": "flake8", "status": "Failed", "details": []}

            file_path, line_number, code, message = flake8_match.groups()
            current_hook["details"].append(
                {
                    "file": file_path,
                    "line": int(line_number),
                    "code": code,
                    "message": message,
                }
            )
            continue
    if current_hook:
        issues.append(current_hook)

    return {"issues": issues}


def save_issues_to_json(
    parsed_data: Dict[str, List[Dict[str, Any]]], filepath: str = "issues.json"
) -> None:
    """Save parsed issues to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(parsed_data, f, indent=4)
