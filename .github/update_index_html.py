import os
import re
import json
import copy
import shutil

# noinspection PyPackageRequirements
from bs4 import BeautifulSoup

INDEX_FILE = "grypi/index.html"
TEMPLATE_FILE = "grypi/pkg_template.html"

VERSION_PATTERN = re.compile(
    "^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    "(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def normalize(name):
    """ From PEP503 : https://www.python.org/dev/peps/pep-0503/ """
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_metadata(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def print_args(args):
    print("\n--- Arguments detected from issue ---\n")
    for arg_name, arg_value in args.items():
        print("\t{} : {}".format(arg_name, arg_value))
    print("\n")


def check_args(args, must_have):
    for name in must_have:
        if name not in args:
            raise ValueError("Couldn't find argument {}".format(name))
        if args[name].strip() == "":
            raise ValueError("Argument {} is empty. Please specify it".format(name))


def package_exists(package_name, html=None):
    if html is None:
        with open(INDEX_FILE) as html_file:
            html = BeautifulSoup(html_file, "html.parser")

    package_ref = package_name + "/"
    for anchor in html.find_all("a"):
        if anchor["href"] == package_ref:
            return True
    return False


def register(args):
    check_args(args, ['package name', 'version', 'author', 'short description', 'long description', 'homepage', 'link'])
    print_args(args)

    with open(INDEX_FILE) as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    n_package_name = normalize(args["package name"])

    if package_exists(n_package_name, html=soup):
        raise ValueError("Package {} seems to already exists".format(n_package_name))

    # Create a new anchor element for our new package
    last_anchor = soup.find_all("a")[-1]  # Copy the last anchor element
    new_anchor = copy.copy(last_anchor)
    new_anchor["href"] = "{}/".format(n_package_name)
    new_anchor.contents[0].replace_with(args["package name"])
    spans = new_anchor.find_all("span")
    spans[1].string = args["version"]  # First span contain the version
    spans[2].string = args["short description"]  # Second span contain the short description

    # Add it to our index and save it
    last_anchor.insert_after(new_anchor)
    with open(INDEX_FILE, "wb") as index:
        index.write(soup.prettify("utf-8"))

    # Then get the template, replace the content and write to the right place
    with open(TEMPLATE_FILE) as temp_file:
        template = temp_file.read()

    template = template.replace("_package_name", args["package name"])
    template = template.replace("_version", args["version"])
    template = template.replace("_link", "{}#egg={}-{}".format(args["link"], n_package_name, args["version"]))
    template = template.replace("_homepage", args["homepage"])
    template = template.replace("_author", args["author"])
    template = template.replace("_long_description", args["long description"])

    package_index = os.path.join("grypi", n_package_name)
    os.mkdir(package_index)
    package_index = os.path.join(package_index, "index.html")

    with open(package_index, "w") as f:
        f.write(template)


def update(args):
    check_args(args, ["package name", "new version", "link for the new version"])
    print_args(args)
    with open(INDEX_FILE) as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    n_package_name = normalize(args["package name"])

    if not package_exists(n_package_name, html=soup):
        raise ValueError("Package {} seems to not exists".format(n_package_name))

    # Change the version in the main page
    anchor = soup.find("a", attrs={"href": "{}/".format(n_package_name)})
    spans = anchor.find_all("span")
    spans[1].string = args["new version"]
    with open(INDEX_FILE, "wb") as index:
        index.write(soup.prettify("utf-8"))

    # Change the package page
    index_file = os.path.join(f"grypi/{n_package_name}/index.html")
    with open(index_file) as html_file:
        soup = BeautifulSoup(html_file, "html.parser")

    # Create a new anchor element for our new version
    last_anchor = soup.find_all("a")[-1]  # Copy the last anchor element
    new_anchor = copy.copy(last_anchor)
    new_anchor["href"] = "{}#egg={}-{}".format(args["link for the new version"], n_package_name, args["new version"])

    # Add it to our index
    last_anchor.insert_after(new_anchor)

    # Change the latest version
    soup.html.body.div.section.find_all("span")[1].contents[0].replace_with(args["new version"])

    # Save it
    with open(index_file, "wb") as index:
        index.write(soup.prettify("utf-8"))


def delete(args):
    check_args(args, ["package name"])
    print_args(args)

    with open(INDEX_FILE) as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    n_package_name = normalize(args["package name"])

    if not package_exists(n_package_name, html=soup):
        raise ValueError("Package {} seems to not exists".format(n_package_name))

    # Remove the package directory
    shutil.rmtree(n_package_name)

    # Find and remove the anchor corresponding to our package
    anchor = soup.find("a", attrs={"href": "{}/".format(n_package_name)})
    anchor.extract()
    with open(INDEX_FILE, "wb") as index:
        index.write(soup.prettify("utf-8"))


def main():

    # Get the context from the environment variable
    context = json.loads(os.environ["GITHUB_CONTEXT"])
    repo_name = context["repository"].split("/")[-1]
    tag_name = context["event"]["ref"].split("/")[-1]

    # dry_run = False
    # if not VERSION_PATTERN.match(tag_name):
    #     dry_run = True

    metadata_file = f"template/metadata.json"
    metadata = parse_metadata(metadata_file)

    # TODO: Fix how the tag is gathered
    args = dict()
    args["version"] = args["new version"] = tag_name
    args["package name"] = repo_name
    args["short description"] = args["long description"] = metadata.get("description", "")
    args["homepage"] = f"https://github.com/{context['repository']}"
    args["link"] = args["link for the new version"] = f"git+{args['homepage']}.git@{args['version']}"
    args["author"] = metadata.get("author", "")

    if not package_exists(repo_name):
        register(args)
    else:
        update(args)


if __name__ == "__main__":
    main()
