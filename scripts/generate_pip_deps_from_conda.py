#!/usr/bin/env python
# This script is a modified copy of generate_pip_deps_from_conda.py from
# pandera, originally from pandas. The original copyright information is
# below.

# This script is a modified copy of generate_pip_deps_from_conda.py from
# pandas and is distributed under the terms of the BSD 3 License that can be
# found at: https://github.com/pandas-dev/pandas/blob/master/LICENSE
"""
Convert the conda environment.yml to the pip requirements-dev.txt,
or check that they have the same packages (for the CI)

Usage:

    Generate `requirements-dev.txt`
    $ ./generate_pip_deps_from_conda.py

    Compare and fail (exit status != 0) if `requirements-dev.txt` has not been
    generated with this script:
    $ ./generate_pip_deps_from_conda --compare
"""
import argparse
import os
import sys
import difflib
import copy

from packaging.requirements import Requirement
import yaml

EXCLUDE = {"python"}
RENAME = {"pytables": "tables"}


def conda_package_to_pip(package: str) -> str:
    """
    Convert a conda package to its pip equivalent.

    In most cases they are the same, those are the exceptions:
    - Packages that should be excluded (in `EXCLUDE`)
    - Packages that should be renamed (in `RENAME`)
    """
    package = Requirement(package)
    package_name = package.name

    if package_name in EXCLUDE:
        return ''

    if package_name in RENAME:
        package = copy.copy(package)
        package.name = RENAME[package_name]

    return str(package)


def main(conda_fname, pip_fname, compare=False):
    """
    Generate the pip dependencies file from the conda file, or compare that
    they are synchronized (``compare=True``).

    Parameters
    ----------
    conda_fname : str
        Path to the conda file with dependencies (e.g. `environment.yml`).
    pip_fname : str
        Path to the pip file with dependencies (e.g. `requirements-dev.txt`).
    compare : bool, default False
        Whether to generate the pip file (``False``) or to compare if the
        pip file has been generated with this script and the last version
        of the conda file (``True``).

    Returns
    -------
    bool
        True if the comparison fails, False otherwise
    """
    with open(conda_fname) as conda_fd:
        deps = yaml.safe_load(conda_fd)["dependencies"]

    pip_deps = []
    for dep in deps:
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            if conda_dep:
                pip_deps.append(conda_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            pip_deps += dep["pip"]
        else:
            raise ValueError(f"Unexpected dependency {dep}")

    fname = os.path.split(conda_fname)[1]
    header = (
            "# This file is auto-generated from %s, do not modify.\n"
            "# See that file for comments about the need/usage of "
            "each dependency.\n\n" % fname
    )
    pip_content = header + "\n".join(pip_deps)

    if compare:
        diff_header = (
                "Comparison between the regenerated `requirements-dev.txt` "
                "and the on-disk version:\n"
        )
        with open(pip_fname) as pip_fd:
            lhs = pip_content.splitlines(keepends=False)
            rhs = pip_fd.read().splitlines(keepends=False)
            diff = list(difflib.unified_diff(lhs, rhs))

            if len(diff) > 0:
                print(diff_header)
                for line in diff:
                    print(line)

                return True

            return False
    else:
        with open(pip_fname, "w") as pip_fd:
            pip_fd.write(pip_content)
        return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="convert (or compare) conda file to pip"
    )
    argparser.add_argument(
        "--compare",
        action="store_true",
        help="compare whether the two files are equivalent",
    )
    argparser.add_argument(
        "--azure",
        action="store_true",
        help="show the output in azure-pipelines format",
    )
    args = argparser.parse_args()

    repo_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    res = main(
        os.path.join(repo_path, "environment.yml"),
        os.path.join(repo_path, "requirements-dev.txt"),
        compare=args.compare,
    )
    if res:
        msg = (
                "`requirements-dev.txt` has to be generated with `%s` after "
                "`environment.yml` is modified.\n" % sys.argv[0]
        )
        if args.azure:
            msg = f"##vso[task.logissue type=error;sourcepath=requirements-dev.txt]{msg}"
        sys.stderr.write(msg)
    sys.exit(res)
