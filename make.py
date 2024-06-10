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

    intro = (
        "# MRI Tools\n"
        "![license](https://img.shields.io/github/license/abdrysdale/mri-tools.svg)\n\n"
        "A collection of free and open-source software software tools for use in MRI.\n"
        "Free is meant as in free beer (gratis) and freedom (libre).\n\n"
        "To add a project edit the repos.toml file and submit a pull request.\n"
        "Repositories are stored in the toml file in the format:\n\n"
        "```toml\n"
        "[repo-name]\nlanguages = [\"repo-lang-1\", \"repo-lang-2\"]\nlink = \"repo-link\"\n"
        "license = \"repo-license\"\ndescription = \"A short description about the repo\"\n"
        "tags = [\"repo-tag-1\", \"repo-tag-2\"]\n"
        "```\n\n"
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
            

    mdrend = (
        "## Stats\n"
        f"- Total repos: {len(repos)}\n"
        f"- Languages:\n\n{_rend_table("Language", languages)}\n"
        f"- Tags:\n\n{_rend_table("Tag", tags)}\n"
        f"- Licenses:\n\n{_rend_table("Licence", licenses)}\n"
    )

    def _get_repo_str(repo, k):
        _out = (
            f"- [{k}]({repo["link"]})\n"        # Repo name + link
            f">- Languages: {", ".join(repo["languages"])}\n" # Languages
            f">- License: {repo["license"]}\n"     # License
            f">- Tags: {", ".join(repo["tags"])}\n"           # Tags
            f">- {repo["description"].title()}\n\n"  # Description
        )
        return _out

    def _toc_item(key):
        return f"[{key}](#{key})"

    stats = {"tags": tags, "languages": languages}
    toc = f"## Table of Contents\n- {_toc_item("stats")}\n"
    for section in ("tags", "languages"):

        toc = toc + f"- {_toc_item(section)}\n"
        mdrend = mdrend + f"\n\n## {section.title()}\n"

        for f in stats[section].keys():

            toc = toc + f"\t- {_toc_item(f)}\n"
            mdrend = mdrend + f"### {f.title()} <a name=\"{f}\"></a>\n"

            for k in repos:

                repo = repos[k]

                if f in repo[section]:
                    mdrend = mdrend + _get_repo_str(repo, k)

    mdrend = intro + toc + "\n" + mdrend

    with open(out, "w") as fp:
        fp.write(mdrend)

    return True
                
if __name__ == "__main__":

    repos = get_repos_from_conf("repos.toml")
    print(f"Repos okay?:\t{check_repos_format(repos)}")
    make(repos)
