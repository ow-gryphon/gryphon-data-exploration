import os
import re
import json
from pathlib import Path

INDEX_FILE = "grypi/index.html"
VERSION_PATTERN = re.compile(
    "^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    "(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def get_index_metadata_path(package_name):
    return Path("grypi") / package_name / "metadata.json"


def read_metadata(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Get the context from the environment variable
    context = json.loads(os.environ["GITHUB_CONTEXT"])
    repo_name = context["repository"].split("/")[-1]
    tag_name = context["event"]["ref"].split("/")[-1]

    dry_run = False
    if not VERSION_PATTERN.match(tag_name):
        dry_run = True

    new_metadata_path = f"template/metadata.json"
    new_metadata = read_metadata(new_metadata_path)
    new_metadata["version"] = tag_name

    index_metadata_path = get_index_metadata_path(repo_name)

    try:
        index_metadata = read_metadata(index_metadata_path)

    except FileNotFoundError:
        metadata = [new_metadata]

        with open(new_metadata_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata))

    else:
        with open(new_metadata_path, "r+", encoding="utf-8") as f:
            metadata = json.load(f)
            
            if type(index_metadata) != list:

                # if the metadata is not in the list format yet, convert it
                new_format = []
                for version, data in index_metadata.items():
                    data["version"] = version
                    new_format.append(data)
                index_metadata = new_format

            index_metadata.append(new_metadata)

            f.seek(0)
            f.write(json.dumps(index_metadata))
            f.truncate()

    if dry_run:
        raise RuntimeError(f"Running dry. Not copying the files to the definitive place. "
                           f"Version name not valid: {tag_name}.")

if __name__ == "__main__":
    main()
