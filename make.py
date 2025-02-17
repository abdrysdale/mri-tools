#! /usr/bin/env python

"""Generates the README from urls.toml."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import tomllib
import urllib.request
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def load_repo_info_from_file(
    name: str, path: str = "repos.json",
) -> dict | None:
    """Load repository information from the path."""
    path = Path(path)
    try:
        with path.open("rb") as fp:
            return json.load(fp)[name]
    except (FileNotFoundError, KeyError):
        return None


def update_repo_info_to_file(
    name: str, info: dict, path: str = "repos.json",
) -> bool:
    """Update the repos.json file."""
    try:
        with Path(path).open("rb") as fp:
            repos_data = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        repos_data = {}
    repos_data[name] = info
    with Path(path).open("w") as fp:
        json.dump(repos_data, fp)
    return True


def maybe_get_repo_info_from_url(
        url: str,
        api: str = "https://api.github.com/repos",
        repo_path: str = "repos.json",
        token: str | None = None,
) -> dict:
    """Get the repository information from the URL.

    Args:
        url : URL to the project.
        api : Root url for the API request.
            Defaults to "https://api.github.com/repos"
        repo_path : Fall back json file for when url fails.
        token : GitHub Auth Token.

    Returns:
        info : Repository information.

    """
    owner, repo = url.split("/")[-2:]
    api_url = f"{api}/{owner}/{repo}"

    request = urllib.request.Request(api_url)
    if token is not None:
        request.add_header("Authorization", f"token {token}")

    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as err:
        logger.warning("%s: %s", url, err)
        return load_repo_info_from_file(repo, repo_path)
    response_data = response.read().decode("utf-8")
    data = json.loads(response_data)
    info = {
        "name": data["name"],
        "description": data["description"],
        "link": url,
        "languages": (
            [data["language"]]
            if isinstance(data["language"], str)
            else data["language"]
        ),
        "license": (
            "None"
            if data["license"] is None
            else data["license"]["name"]
        ),
        "tags": data["topics"],
        "forks": data["forks"],
        "open_issues": data["open_issues"],
        "watchers": data["watchers"],
        "updated_at": datetime.datetime.fromisoformat(
            data["updated_at"],
        ).strftime("%Y-%m-%d"),
    }
    update_repo_info_to_file(repo, info, repo_path)
    return info


def get_repos_from_conf(
        url_path: str, repo_path: str = "repos.json", token: str | None = None,
) -> list[dict]:
    """Return a dictionary of the repos.

    Args:
        url_path : Path to the toml file containing the urls.
        repo_path : Path to the json file containing the repo information.
            Defaults to "repos.json" - used if url fetching fails.
        token : GitHub Auth token.

    Returns:
        repos (list[dict]) : List containing the repo information.

    """
    with Path(url_path).open("rb") as fp:
        urls = tomllib.load(fp)["urls"]
    repos = [maybe_get_repo_info_from_url(url, token=token) for url in urls]
    return [r for r in repos if r]


def count_entries_in_field(repos: list[dict], field: str) -> dict:
    """Get the number of occurances of each entry within a field.

    Args:
        repos : List containg the repo information.
        field : Field to count.

    Returns:
        count_dict (dict) : Dictionary of the field occurances.

    """
    cnt = Counter()
    for repo in repos:
        vals = repo[field]
        if isinstance(vals, (list, tuple)):
            for v in vals:
                cnt[v.lower()] += 1
        else:
            cnt[vals.lower()] += 1
    return dict(cnt.most_common())

def make(
    repos: list[dict], out: str = "README.md", toc_thresh: int = 2,
) -> bool:
    """Render the markdown document from the repos toml file.

    Args:
        repos : List containg the repo information.
        out : Output file to render to.
                Defaults to "README.md"
        toc_thresh : Number of entries needed to be considered in the
            table of contents. Defaults to 2.

    Returns:
        return_val (bool) : True if sucessful, False otherwise.

    """
    intro = (
        "# MRI Tools\n"
        "![license](https://img.shields.io/github/license/abdrysdale/mri-tools.svg)\n\n"
        "A collection of free and open-source software software tools for use in MRI.\n"
        "Free is meant as in free beer (gratis) and freedom (libre).\n\n"
        "To add a project, add the project url to the `urls.toml` file.\n\n"
    )

    languages = count_entries_in_field(repos, "languages")
    tags = count_entries_in_field(repos, "tags")
    licenses = count_entries_in_field(repos, "license")

    def _rend_table(col: str, cnt_dict: dict) -> str:
        _out = (
            f"| {col} | Count |\n"
            "|---|---|\n"
        )

        for k in cnt_dict:
            _out = _out + f"| {k} | {cnt_dict[k]} |\n"

        return _out

    mdrend = (
        "## Stats\n"
        f"- Total repos: {len(repos)}\n"
        f"- Languages:\n\n{_rend_table("Language", languages)}\n"
        f"- Tags:\n\n{_rend_table("Tag", tags)}\n"
        f"- Licenses:\n\n{_rend_table("Licence", licenses)}\n"
    )

    def _get_repo_str(repo: dict, k: str) -> str:
        return (
            f"- [{k}]({repo["link"]})\n"                        # Repo name/link
            f">- {repo["description"]}\n\n"                     # Description
            f">- License: {repo["license"]}\n"                  # License
            ">- Languages: "                                    # Languages
            f"{", ".join(["`"+l.title()+"`" for l in repo["languages"]])}\n"
            f">- Tags: {", ".join(repo["tags"])}\n"             # Tags
            f">- Forks:\t{repo["forks"]} \n"                    # Forks
            f">- Issues:\t{repo["open_issues"]}\n"              # Open issues
            f">- Watchers:\t{repo["watchers"]}\n"               # Watchers
            f">- Last updated: {repo["updated_at"]}\n\n"        # Last Updated
        )


    def _toc_item(key: str) -> str:
        return f"[{key}](#{key})"

    stats = {"tags": tags, "languages": languages}
    toc = f"## Table of Contents\n- {_toc_item("stats")}\n"
    for section in ("tags", "languages"):
        toc = toc + f"- {_toc_item(section)}\n"
        mdrend = mdrend + f"\n\n## {section.title()}\n"
        for f, cnt in stats[section].items():
            if cnt >= toc_thresh:
                toc = toc + f"\t- {_toc_item(f)}\n"
            mdrend = mdrend + f'### {f.title()} <a name="{f}"></a>\n'
            for repo in repos:
                if f in repo[section]:
                    mdrend = mdrend + _get_repo_str(repo, repo["name"])

    mdrend = intro + toc + "\n" + mdrend

    with Path(out).open("w") as fp:
        fp.write(mdrend)

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make the README",
    )

    parser.add_argument(
        "-t",
        "--token",
        default=None,
        type=str,
        help="GitHub Auth Token.",
    )
    args = parser.parse_args()

    with Path("urls.toml").open("rb") as fp:
        ttl_urls = len(tomllib.load(fp)["urls"])
    repos = get_repos_from_conf("urls.toml", token=args.token)
    print(f"Repos found:\t{len(repos)}/{ttl_urls}")  # noqa: T201

    make(repos)
