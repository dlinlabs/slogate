"""Print shell assignments for a SUT config so run scripts never parse JSON in bash."""

import json
import shlex
import sys


def array(name: str, values: list[str]) -> str:
    return f"{name}=(" + " ".join(shlex.quote(str(v)) for v in values) + ")"


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as handle:
        config = json.load(handle)
    print(f"SUT_NAME={shlex.quote(config['name'])}")
    print(f"MODEL_ID={shlex.quote(config['model']['id'])}")
    print(f"MODEL_REVISION={shlex.quote(config['model']['revision'])}")
    print(array("SERVE_ARGS", config["serve_args"]))
    print(array("ENGINE_REQUIREMENTS", config["engine"]["requirements"]))
    print(array("ENGINE_PIP_ARGS", config["engine"].get("pip_args", [])))


if __name__ == "__main__":
    main()
