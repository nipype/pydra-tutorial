---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Intro to Pydra

+++

Pydra is a lightweight, Python 3.7+ dataflow engine. While it originated within the neuroimaging community, its versatile design makes it suitable as a general-purpose engine to facilitate analytics across various scientific fields.


You can discover a more in-depth explanation of the concept behind Pydra here. TODO-LINK

+++

In this tutorial you will create and execute your first `Task`s from standard Python functions and shell commands. 
You'll also construct basic `Workflow`s that link multiple tasks together. Furthermore, you'll have the opportunity to produce `Task`s and `Workflow`s capable of automatically running for multiple inputs values.

+++

**Before going to the main notebooks, let's check if pydra is properly installed.** If you have any issues running the following cell, please revisit the Installation section. TODO-LINK

```{code-cell} ipython3
import pydra
```

### Additional notes

+++

At the beginning of each tutorial you will see:
```
import nest_asyncio
nest_asyncio.apply()
```
This is run because both *Jupyter* and *Pydra* use `asyncio` and in some cases you can see `RuntimeError: This event loop is already running` if `nest_asyncio` is not used. **This part is not needed if Pydra is used outside the Jupyter environment.**
