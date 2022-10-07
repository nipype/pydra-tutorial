---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Intro to Pydra

+++

Pydra is a lightweight, Python 3.7+ dataflow engine for computational graph construction, manipulation, and distributed execution.
Designed as a general-purpose engine to support analytics in any scientific domain; created for [Nipype](https://github.com/nipy/nipype), and helps build reproducible, scalable, reusable, and fully automated, provenance tracked scientific workflows.
The power of Pydra lies in ease of workflow creation
and execution for complex multiparameter map-reduce operations, and the use of global cache.

Pydra's key features are:
- Consistent API for Task and Workflow
- Splitting & combining semantics on Task/Workflow level
- Global cache support to reduce recomputation
- Support for execution of Tasks in containerized environments

+++

## Pydra computational objects - Tasks
There are two main types of objects in *pydra*: `Task` and `Workflow`, that is also a type of `Task`, and can be used in a nested workflow.
![nested_workflow.png](../figures/nested_workflow.png)



**These are the current `Task` implemented in Pydra:**
- `Workflow`: connects multiple `Task`s withing a graph
- `FunctionTask`: wrapper for Python functions
- `ShellCommandTask`: wrapper for shell commands
    - `ContainerTask`: wrapper for shell commands run within containers
      - `DockerTask`: `ContainerTask` that uses Docker
      - `SingularityTask`: `ContainerTask` that uses Singularity

+++

## Pydra Workers
Pydra supports multiple workers to execute `Tasks` and `Workflows`:
- `ConcurrentFutures`
- `SLURM`
- `Dask` (experimental)

+++

**Before going to next notebooks, let's check if pydra is properly installed**

```{code-cell}
import pydra
```
