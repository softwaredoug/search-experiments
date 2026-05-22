# Agentic Filesystem Search Tools

(This set of tools is an extension of the [agentic toolset](docs/agentic_prd.md) adding tools to search the file system).

Modern agents can use coding tools to search filesystems surprisingly well. So a set of tools (grep, etc) can be used in an 
agentic strategy to search a virtual filesystem

In this document, we describe that virtual file system, built on top of the data frame.


## The virtual filesystem

The corpus we load is a pandas dataframe. We pretend this dataframe is a virtual file system. To do this, we
add special columns to represent the path of the item, including a filename. Each row in the dataframe corresponds
to a "file".

path: by default, the document title plus ID as a SLUG. IE a document's title is "Red Shoes" and its ID is 123, then the
path would be /red-shoes-123.txt.

contents: The contents of the file would include a title + description + doc_id, as follows:

```
# <Title Text> (ID: <DocID>) 

<Description Text>
```

```
# Red Shoes (ID: 1234)

These are the best red shoes you'll ever find. They're super comfy and stylish.
```

## Tools

Its expected in an agentic strategy we would have the following tools as python functions, similar to other tools in
docs/agentic_prd.md


- `ls`: list files in a directory that match a glob. Can be used to navigate the virtual file system.
- `grep`: given glob + regex, search the contents of the files and return matching files. This is the main search tool.
- `cat`: given a file path, return the contents of the file. This allows the agent to read the contents of a file it
  found with grep or ls.

Roughly their signature looks like:

```python
def ls(path: str, glob: str, max_results=50) -> List[str]:
    """List files in a directory matching the glob, at most 50 results. Returns a list of paths."""
    pass
```

```python
def grep(pattern: str, glob: str, num_results=50) -> List[Dict[str, str]]:
    """Search for a pattern in files matching the glob, at most 50 results.

    Returns a list of dicts like {"path": <path>, "snippet": <snippet>}.
    """
    pass
```

```python
def cat(path: str) -> str:
    """Return the contents of a file as a string."""
    pass
```


### Tool setup

Its expected when these tools are created, through a factory function, that the dataframe will be modified to include a
`path` column (and a `contents` column) as detailed here. This is synonomous to "indexing" but should be done with pandas
code just creating the columns.

If somehow `path` already exists before tool setup starts, throw an error.

### Plug into existing tool plumbing

There's existing tool registry plumbing in this code these tools plug into.

### Not all tools listed

It's ok, if weird, if only a subset of these tools are listed in the config.

### WANDS filesystem tools

For WANDS, dataset-specific tools use the same tool behavior but construct paths with
category and subcategory prefixes:

```
/<category>/<subcategory>/<title-slug>-<id>.txt
```

When subcategory is missing, the path becomes:

```
/<category>/<title-slug>-<id>.txt
```

When both category and subcategory are missing, the path remains at root.

Tool names:

- `ls_wands`
- `grep_wands`
- `cat_wands`

### Raw filesystem / bash tool

If the tool `bash` is used, then direct bash commands are allowed.

Instead of searching a pandas dataframe, you should write the corpus to the actual file system. 

It should be written to 

<search experimetns root>/filesystem/<dataset_name>/

Here <search experimetns root> is the data / working directory (ie ~/.search-experiments) for this repo

If this exists, do not regenerate it.

Now the agent can invoke `bash` commands to search the file system directly.

Tell the agent the directory the data is stored and that's where it should search. 

### Docker container with service 

Start a docker container with a small service that takes a bash command, executes it, and returns the results. This is a
more secure way to allow bash commands without giving the agent direct access to the file system.

Just use built in http server libraries in python to do this. The service should be started when the tool is created and
stopped when the tool is destroyed.

It should basically mount the data directory and have an endpoint like /execute that takes a bash command, executes it,
and returns the results.

The docker container should mount the <search experimetns root>/filesystem/<dataset_name>/ directory to /corpus in the
container, and the service should execute bash commands in that directory.

### Dataset specific

Note if 

bash_wands

is use, then the WANDS data should be written to the filesystem with the category/subcategory structure as described
above. The service should also be aware of this structure and execute commands in the correct directory.
