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

contents: The contents of the file would include a title + description, as follows:

```
Title: Red Shoes

Description: These are the best red shoes you'll ever find. They're super comfy and stylish.
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
