"""Nox sessions."""
# This script is a modified copy of noxfile.py from
# pandera and is distributed under the terms of the BSD 3 License that can be
# found at: https://github.com/pandera-dev/pandera/blob/master/LICENSE.txt

import configparser
import os
import shutil
import sys
from typing import Dict, List, Union

import dunamai
import nox
from nox import Session
from packaging.requirements import Requirement

nox.options.sessions = (
    "requirements",
    "black",
    "isort",
    "lint",
    "mypy",
    "tests",
    # "docs", # FIXME: re-enable documentation generation
)

DEFAULT_PYTHON = "3.10"
PYTHON_VERSIONS = ["3.10"]

PACKAGE = "connectfour"

SOURCE_PATHS = "src", "test", "noxfile.py"
REQUIREMENT_PATH = "requirements-dev.txt"
RENAME = {"tables": "pytables"}
ALWAYS_USE_PIP = [
    "pylint-junit",
    # TODO: place any pip-only dependencies here and remove the following
    "pandera",
]  # ["furo"]  # FIXME: re-enable with documentation generation

CI_RUN = os.environ.get("CI") == "true"
if CI_RUN:
    print("Running on CI")
else:
    print("Running locally")

LINE_LENGTH = 79


def parse_requirements(
    reqs: Union[str, List[str]], keep_markers: bool = False
) -> List[Requirement]:
    """
    A quick-and-dirty requirements.txt parser.

    Args:
        reqs: a list of strings representing requirements or
            the contents of the requirements.txt file.
        keep_markers: whether to remove markers like "python_version < '3.8'"

    Warnings:
        This does not handle line continuations properly, and comments are
        only taken into account if the # is the first character
    """
    if isinstance(reqs, str):
        # split string by line
        reqs = reqs.splitlines()
        reqs = [line for line in reqs if not line.startswith("#")]

    result = [Requirement(r) for r in reqs if r]

    if keep_markers:
        return result

    # nullify markers if any
    for r in result:
        r.marker = None

    return result


def _build_setup_requirements() -> Dict[str, List[Requirement]]:
    """Load requirements from setup.cfg."""
    config = configparser.ConfigParser()
    config.read("setup.cfg", encoding="utf-8")

    # extract core
    reqs = {
        "core": parse_requirements(
            config.get("options", "install_requires", fallback="")
        ),
    }

    if config.has_section("options.extras_require"):
        for k, val in config.items("options.extras_require"):
            reqs[k] = parse_requirements(val)
    return reqs


def _build_dev_requirements() -> List[Requirement]:
    """Load requirements from file."""
    with open(REQUIREMENT_PATH, "rt", encoding="utf-8") as req_file:
        dev_req = parse_requirements(req_file.read())

    # install keyrings.alt if running on CI
    if CI_RUN:
        dev_req.extend(parse_requirements("keyrings.alt"))

    return dev_req


SETUP_REQUIREMENTS: Dict[str, List[Requirement]] = _build_setup_requirements()
DEV_REQUIREMENTS: List[Requirement] = _build_dev_requirements()


def _requirement_to_dict(reqs: List[Requirement]) -> Dict[str, str]:
    """Return a dict {PKG_NAME:PIP_SPECS}."""
    return {req.name: str(req) for req in reqs}


def _build_requires() -> Dict[str, Dict[str, str]]:
    """Return a dictionary of requirements {EXTRA_NAME: {PKG_NAME:PIP_SPECS}}.

    Adds fake extras "core" and "all".
    """
    extras = {
        extra: reqs
        for extra, reqs in SETUP_REQUIREMENTS.items()
        if extra not in ("core", "dev", "all")
    }
    extras["all"] = DEV_REQUIREMENTS

    optionals = [
        req.name
        for extra, reqs in extras.items()
        for req in reqs
        if extra != "all"
    ]
    requires = {"all": _requirement_to_dict(extras["all"])}
    requires["core"] = {
        pkg: specs
        for pkg, specs in requires["all"].items()
        if pkg not in optionals
    }
    requires.update(  # add extras
        {
            extra_name: {**_requirement_to_dict(pkgs), **requires["core"]}
            for extra_name, pkgs in extras.items()
            if extra_name != "all"
        }
    )

    # convert from Requirement to str
    mapped_requires = {}
    for extra, reqs in requires.items():
        mapped_requires[extra] = {name: str(req) for name, req in reqs.items()}

    return mapped_requires


REQUIRES: Dict[str, Dict[str, str]] = _build_requires()

CONDA_ARGS = [
    "--channel=conda-forge",
    "--update-specs",
]


def pip_package_to_conda(package: str) -> Requirement:
    """
    Convert a pip package to its conda equivalent.

    In most cases they are the same, those are the exceptions:
    - Packages that should be renamed (in `RENAME`)
    """
    _package = Requirement(package)
    if _package.name in RENAME:
        _package.name = RENAME[_package.name]

    return _package


def conda_install(session: Session, *args):
    """Use mamba to install conda dependencies."""
    renamed_for_conda = [str(pip_package_to_conda(arg)) for arg in args]
    run_args = [
        "install",
        "--yes",
        *CONDA_ARGS,
        "--prefix",
        session.virtualenv.location,  # type: ignore
        *renamed_for_conda,
    ]

    # By default, all dependencies are re-installed from scratch with each
    # session. Specifying external=True allows access to cached packages, which
    # decreases runtime of the test sessions.
    try:
        session.run(
            *["mamba", *run_args],
            external=True,
        )
    # pylint: disable=broad-except
    except Exception:
        session.run(
            *["conda", *run_args],
            external=True,
        )


def install(session: Session, *args: str, force_pip: bool = False):
    """Install dependencies in the appropriate virtual environment
    (conda or virtualenv) and return the type of the environmment."""
    if (
        isinstance(session.virtualenv, nox.virtualenv.CondaEnv)
        and not force_pip
    ):
        session.warn("using conda installer")
        conda_install(session, *args)
    else:
        session.warn("using pip installer")
        session.install(*args)


def install_from_requirements(session: Session, *packages: str) -> None:
    """
    Install dependencies, respecting the version specified in requirements.
    """
    for package in packages:
        try:
            spec = REQUIRES["all"][package]
        except KeyError:
            raise ValueError(
                f"{package} cannot be found in {REQUIREMENT_PATH}."
            ) from None
        install(session, spec)


def install_extras(
    session: Session,
    extra: str = "core",
    force_pip=False,
) -> None:
    """Install dependencies."""
    specs = [
        spec
        for key, spec in REQUIRES[extra].items()
        if key not in ALWAYS_USE_PIP
    ]

    always_pip_specs = [
        spec for key, spec in REQUIRES[extra].items() if key in ALWAYS_USE_PIP
    ]

    install(session, *specs, force_pip=force_pip)

    # always use pip for these packages
    session.install(*always_pip_specs)
    session.install("-e", ".", "--no-deps")


def _generate_pip_deps_from_conda(
    session: Session, compare: bool = False
) -> None:
    args = ["scripts/generate_pip_deps_from_conda.py"]
    if compare:
        args.append("--compare")
    session.run("python", *args)


@nox.session(python=PYTHON_VERSIONS)
def requirements(session: Session) -> None:  # pylint:disable=unused-argument
    """Check that setup.py requirements match requirements-dev.txt"""
    install(session, "pyyaml", "packaging")
    try:
        _generate_pip_deps_from_conda(session, compare=True)
    except nox.command.CommandFailed as err:
        _generate_pip_deps_from_conda(session)
        print(f"{REQUIREMENT_PATH} has been re-generated ✨ 🍰 ✨")
        raise err

    # Comparisons between requirements are unsupported, so we convert to str
    dev_req = {str(_) for _ in DEV_REQUIREMENTS}

    ignored_pkgs = {"black"}
    mismatched = []
    for extra, reqs in SETUP_REQUIREMENTS.items():
        if extra not in EXTRA_NAMES:
            continue

        for req in reqs:
            if req.name not in ignored_pkgs and str(req) not in dev_req:
                mismatched.append(f"{extra}: {req.name} ({req})")

    if mismatched:
        print(
            f"Packages {mismatched} defined in setup.cfg "
            f"do not match {REQUIREMENT_PATH}."
        )
        print(
            "Modify environment.yml, "
            + f"then run 'nox -s requirements' to generate {REQUIREMENT_PATH}"
        )
        sys.exit(1)


@nox.session(python=DEFAULT_PYTHON)
def black(session: Session) -> None:
    """Check black style."""
    install_from_requirements(session, "black")
    args = ["--check"] if CI_RUN else session.posargs
    session.run(
        "black",
        f"--line-length={LINE_LENGTH}",
        *args,
        *SOURCE_PATHS,
    )


@nox.session(python=DEFAULT_PYTHON)
def isort(session: Session) -> None:
    """Check isort style."""
    install_from_requirements(session, "isort")
    args = ["--check-only"] if CI_RUN else session.posargs
    session.run(
        "isort",
        f"--line-length={LINE_LENGTH}",
        "--profile=black",
        *args,
        *SOURCE_PATHS,
    )


def _preprocess_version(version: str = None) -> str:
    """Write version file, either by using input or computing from dunamai."""
    if version:
        dunamai.check_version(version, style=dunamai.Style.Pep440)
    else:
        # use dunamai to get the version from git
        _version = dunamai.Version.from_git()
        version = _version.serialize(tagged_metadata=True)

    return version


# pass through to source environment
@nox.session(python=False)
def build(session: Session) -> None:
    """Build package and upload to our private packaging server."""
    if len(session.posargs) > 1:
        session.warn("Build takes at most one argument, the proposed version")
        sys.exit(-1)

    version = None if not session.posargs else session.posargs[0]
    version = _preprocess_version(version)

    if not CI_RUN:
        session.warn("Not running on CI, version string %s unused.", version)
        session.run("python", "-m", "build", "-n")
        return

    session.log("Writing version %s to VERSION.", version)
    with open("VERSION", "w", encoding="utf-8") as f:
        f.writelines([version])

    session.run("python", "-m", "build", "-n")

    # actually push the image; uses keyring to get credentials
    user = os.environ.get("CIRCLECI_PYPI_USER")
    userargs = () if user is None else ("-u", user)
    session.run(
        "twine",
        "upload",
        "--non-interactive",
        *userargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def lint(session: Session) -> None:
    """Lint using pylint."""
    install_extras(session, extra="all")
    args = session.posargs or SOURCE_PATHS

    if session.python == "3.9":
        # https://github.com/PyCQA/pylint/issues/776
        args = ["--disable=unsubscriptable-object", *args]
    session.run("pylint", *args)


@nox.session(python=PYTHON_VERSIONS)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    install_extras(session, extra="all")
    args = session.posargs or SOURCE_PATHS
    session.run("mypy", "--follow-imports=silent", *args, silent=True)


EXTRA_NAMES = [
    extra
    for extra in REQUIRES
    if extra != "all" and not extra.startswith(":python_version")
]


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("extra", EXTRA_NAMES)
def tests(session: Session, extra: str) -> None:
    """Run the test suite."""
    install_extras(session, extra)

    if session.posargs:
        args = session.posargs
    else:
        path = f"test/{extra}/" if extra != "all" else "test"
        args = []
        args += [
            f"--cov={PACKAGE}",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-append",
        ]
        if not CI_RUN:
            args.append("--cov-report=html")
        args.append(path)

    session.run("pytest", *args)


@nox.session(python=PYTHON_VERSIONS)
def docs(session: Session) -> None:
    """Build the documentation."""
    install_extras(session, extra="all")
    session.chdir("docs")

    shutil.rmtree(os.path.join("_build"), ignore_errors=True)
    args = session.posargs or [
        "-v",
        "-v",
        "-W",
        "-E",
        "-b=doctest",
        "source",
        "_build",
    ]
    session.run("sphinx-build", *args)

    # build html docs
    if not CI_RUN and not session.posargs:
        shutil.rmtree(os.path.join("docs", "_build"), ignore_errors=True)
        session.run(
            "sphinx-build",
            "-W",
            "-T",
            "-b=html",
            "-d",
            os.path.join("_build", "doctrees", ""),
            "source",
            os.path.join("_build", "html", ""),
        )
