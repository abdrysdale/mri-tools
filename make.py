#! /usr/bin/env python

"""Generates the README from repos.toml"""

# Python imports
import tomllib
from collections import Counter


def get_repos_from_conf(path: str) -> list[dict]:
    """Returns a dictionary of the repos

    Args:
        path (str) : Path to the toml file.

    Returns:
        repos (list[dict]) : List containing the repo information.
    """

    with open(path, "rb") as fp:
        repos = tomllib.load(fp)
    return repos


def check_repos_format(repos: list[dict]) -> bool:
    """Checks the repos are correctly formatted

    Args:
        repos (list[dict]) : List containg the repo information.

    Returns:
        return_val (bool) : True if sucessful, False otherwise.
    """

    fields = {
        "languages": list,
        "link": str,
        "license": str,
        "description": str,
        "tags": list
    }

    for k in repos:
        repo = repos[k]
        if len(repo.keys()) != len(fields.keys()):
            return False
        for key in repo.keys():
            if key not in list(fields.keys()):
                return False
            if not isinstance(repo[key], fields[key]):
                return False
    return True

def count_entries_in_field(repos: list[dict], field: str) -> dict:
    """Gets the number of occurances of each entry within a field
    
    Args:
        repos (list[dict]) : List containg the repo information.

    Returns:
        count_dict (dict) : Dictionary of the field occurances.
    """

    cnt = Counter()
    for k in repos:
        vals = repos[k][field]
        if isinstance(vals, (list, tuple)):
            for v in vals:
                cnt[v.lower()] += 1
        else:
            cnt[vals.lower()] += 1
    return dict(cnt.most_common())

def make(repos: list[dict], out: str = "README.md") -> bool:
    """Renders the markdown document from the repos toml file

    Args:
        repos (list[dict]) : List containg the repo information.
        out (str, optional) : Output file to render to.
                Defaults to "README.md"

    Returns:
        return_val (bool) : True if sucessful, False otherwise.
    """

    mdrend = (
        "# MRI Tools\n\n"
        "A collection of free and open-source software software tools for use in MRI.\n"
        "Free is meant as in free beer (gratis) and freedom (libre).\n\n"
        "To add a project edit the repos.toml file and submit a pull request.\n"
    )

    languages = count_entries_in_field(repos, "languages")
    tags = count_entries_in_field(repos, "tags")
    licenses = count_entries_in_field(repos, "license")

    def _rend_table(col, cnt_dict):
        _out = (
            f"| {col} | Count |\n"
            "|---|---|\n"
        )

        for k in cnt_dict.keys():
            _out = _out + f"| {k} | {cnt_dict[k]} |\n"

        return _out
            

    mdrend = mdrend + (
        "## Stats\n"
        f"- Total repos: {len(repos)}\n"
        f"- Languages:\n\n{_rend_table("Language", languages)}\n"
        f"- Tags:\n\n{_rend_table("Tag", tags)}\n"
        f"- Licenses:\n\n{_rend_table("Licence", licenses)}\n"
    )

    def _get_repo_str(repo, k):
        _out = (
            f"- [{k}]({repo["link"]})\n"        # Repo name + link
            f">- languages: {repo["languages"]}\n" # Languages
            f">- license: {repo["license"]}\n"     # License
            f">- Tags: {repo["tags"]}\n"           # Tags
            f">- {repo["description"].title()}\n\n"  # Description
        )
        return _out

    mdrend = mdrend + "\n\n## Languages\n"
    for l in languages.keys():

        mdrend = mdrend + f"### {l.title()}\n"
        for k in repos:
            repo = repos[k]
            if l in repo["languages"]:
                mdrend = mdrend + _get_repo_str(repo, k)

    mdrend = mdrend + "\n\n## Tags\n"
    for t in tags.keys():

        mdrend = mdrend + f"### {t.title()}\n"
        for k in repos:
            repo = repos[k]
            if t in repo["tags"]:
                mdrend = mdrend + _get_repo_str(repo, k)

    with open(out, "w") as fp:
        fp.write(mdrend)

    return True
                
if __name__ == "__main__":

    repos = get_repos_from_conf("repos.toml")
    print(f"Repos okay?:\t{check_repos_format(repos)}")
    make(repos)
