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

# Workers in Pydra

Pydra workers are classes that are used to execute `Tasks` and `Workflows`. Pydra currently supports the following workers:
- `SerialWorker`
- `ConcurrentFuturesWorker`
- `SlurmWorker`
- `DaskWorker`
- `SGEWorker`
- `PsijWorker`

## 1. SerialWorker

A worker to execute linearly. 
```
with pydra.Submitter(plugin='serial') as sub:
    sub(wf)
```

## 2. ConcurrentFuturesWorker

A worker to execute in parallel using Python's concurrent futures.
```
with pydra.Submitter(plugin='cf') as sub:
    sub(wf)
```

## 3. SlurmWorker

A worker to execute tasks on SLURM systems.
```
with pydra.Submitter(plugin='slurm') as sub:
    sub(wf)
```

## 4. DaskWorker

A worker to execute in parallel using Dask.distributed.
```
with pydra.Submitter(plugin='dask') as sub:
    sub(wf)
```

## 5. SGEWorker

A worker to execute tasks on SLURM systems.
```
with pydra.Submitter(plugin='sge') as sub:
    sub(wf)
```

## 6. PsijWorker

A worker to execute tasks using PSI/J executors. Currently supported executors are: `local` and `slurm`.
```
with pydra.Submitter(plugin='psij-local') as sub:
    sub(wf)
```
```
with pydra.Submitter(plugin='psij-slurm') as sub:
    sub(wf)
```