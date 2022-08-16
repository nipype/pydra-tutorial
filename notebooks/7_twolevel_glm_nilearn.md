---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 7. Multilevel GLM (from Nilearn)

+++

In this tutorial, we demonstrate how to write pydra tasks for the first and second level GLM in Nilearn. We use the data from [Balloon Analog Risk-taking Task](https://openneuro.org/datasets/ds000001/versions/1.0.0). 
Basic information about this dataset:
- 16 subjects
- 3 runs
- functional scan TR: 2.3 
- num of functional scan: 300

```{code-cell} ipython3
import nest_asyncio
nest_asyncio.apply()
```

## Preparation

Import packages that will be used globally and set up output directory

```{code-cell} ipython3
import os, glob
import datetime
import pydra
from pydra import Workflow
from pydra.engine.specs import File, MultiInputFile, MultiOutputFile
import typing as ty
from pathlib import Path

# get current directory
pydra_tutorial_dir = os.path.dirname(os.getcwd())

# set up output directory
workflow_dir = Path(pydra_tutorial_dir) / 'outputs'
workflow_out_dir = workflow_dir / '7_glm' /'results'

# create the output directory if not exit
os.makedirs(workflow_out_dir, exist_ok=True)
```

## Download the data

[DataLad](http://handbook.datalad.org/en/latest/index.htmlhttp://handbook.datalad.org/en/latest/index.html) is often used in those cases to download data. Here we use its [Python API](http://docs.datalad.org/en/latest/modref.htmlhttp://docs.datalad.org/en/latest/modref.html).

We need the following data: 

1. event information (raw data)
2. preprocessed image data (fmriprep)
3. confounds (fmriprep)

By `api.install`, datalad downloads all symlinks without storing the actual data locally. We can then use `api.get` to get the data we need for our analysis. 
We need to get three types of data from two folders:

1. `*events.tsv` from `rawdata_path`
2. `*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz` from `fmriprep_path`
3. `*desc-confounds_timeseries.tsv` from `fmriprep_path`

```{code-cell} ipython3
fmriprep_path = workflow_out_dir / 'data'
rawdata_path = workflow_out_dir / 'raw_data'
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'rawdata_url': str,
        'fmriprep_url': str,
        'return': {'event_list': MultiOutputFile, 
                   'img_list': MultiOutputFile, 
                   'mask_list': MultiOutputFile, 
                  },
    }
)
def get_data(rawdata_url, fmriprep_url):
    print("Download data...")
    t1 = datetime.datetime.now()
    print(t1)
    import datalad.api as dl
    fmriprep_path = workflow_dir / '7_glm'/ 'data'
    rawdata_path = workflow_dir / '7_glm' / 'raw_data'
    
    # Install datasets to specific datapaths
    dl.install(source=rawdata_url, path=fmriprep_path)
    dl.install(source=fmriprep_url, path=rawdata_path)
    
    # get events.tsv list
    event_list = glob.glob(os.path.join(rawdata_path, '*', 'func', '*events.tsv'))
    event_list.sort()
    # for i in event_list:
    #     dl.get(i, dataset=rawdata_path)
    # get img list
    img_list = glob.glob(os.path.join(fmriprep_path, '*', 'func', '*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'))
    img_list.sort()
    # for i in img_list:
    #     dl.get(i, dataset=fmriprep_path)
    
     # get img list
    mask_list = glob.glob(os.path.join(fmriprep_path, '*', 'func', '*space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'))
    mask_list.sort()
    # for i in mask_list:
    #     dl.get(i, dataset=fmriprep_path)

    t2 = datetime.datetime.now()
    print(t2-t1)
    return event_list, img_list, mask_list
```

## First-Level GLM

We conduct the first level GLM for each run on every subject.

+++

### Get events, preproc_bold, and confounds for each subject

each subject will have a list of three (run) of those files

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'subj_id': int,
        'n_run': int,
        'event_list': list, 
        'img_list': list, 
        'mask_list': list,
        'return': {'subj_id': int, 'subj_events': list, 'subj_imgs':list, 'subj_masks':list},
    }
)
def get_subj_file(subj_id, n_run, event_list, img_list, mask_list):
    t1 = datetime.datetime.now()
    print(f"\nGet subject-{subj_id+1} file...\n")
    # subj_id starts from 0
    start = subj_id*n_run
    end = (subj_id+1)*n_run
    subj_events = event_list[start:end]
    subj_imgs = img_list[start:end]
    subj_masks = mask_list[start:end]
    t2 = datetime.datetime.now()
    print(t2-t1)
    return subj_id, subj_events, subj_imgs, subj_masks
```

### Get the first-level design matrix

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'tr': float,
        'n_scans': int,
        'hrf_model': str,
        'subj_id': int,
        'subj_imgs': list,
        'subj_events':list,
        'return': {'design_matrices': list, 'dm_paths':list},
    }
)
def get_firstlevel_dm(tr, n_scans, hrf_model, subj_id, subj_imgs, subj_events):
    t1 = datetime.datetime.now()
    print(f"\nGet subject-{subj_id+1} firstlevel GLM ...\n")
    import numpy as np
    import pandas as pd
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    # read event file
    events = []
    imgs = []
    for run_event in subj_events:
        event = pd.read_csv(run_event, sep='\t').fillna(0)
        event = event[['onset', 'duration', 'trial_type']]
        events.append(event)
    
    # get list of confounds directly from fmriprepped bold
    confounds = load_confounds_strategy(subj_imgs, denoise_strategy='simple')[0]
    
    frame_times = np.arange(n_scans) * tr
    design_matrices = []
    dm_paths = []
    for index, (ev, conf) in enumerate(zip(events, confounds)):
        design_matrix = make_first_level_design_matrix(frame_times, ev, 
                                                       hrf_model=hrf_model,
                                                       add_regs=conf)          
        
        # make sure all design matrices have the same length of column
        # if you have a block design, this is not needed.
        # 39 = 4(events) + 34(confounds) + 13(drift) + 1(constant)
        assert design_matrix.shape[1] == 52, "This design matrix has the wrong column number"
        # sort the column order alphabetical for contrasts
        design_matrix = design_matrix.reindex(sorted(design_matrix.columns), axis=1)
        dm_path = os.path.join(workflow_out_dir, 'sub-%s_run-%s_designmatrix.csv' % (subj_id+1, index+1))
        design_matrix.to_csv(dm_path, index=None)
        design_matrices.append(design_matrix)
        dm_paths.append(dm_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return design_matrices, dm_paths
```

### Set up the first-level contrasts

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'subj_id': int,
        'design_matrices': list,
        'return': {'contrasts': dict, 'contrast_plot':list},
    }
)
def set_contrast(subj_id, design_matrices):
    t1 = datetime.datetime.now()
    print(f"\nSet firstlevel contrast for subject-{subj_id+1} ...\n")
    
    import pandas as pd
    import numpy as np
    from nilearn.plotting import plot_contrast_matrix
    
    design_matrix = design_matrices[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])
    contrasts = {
        'pumps-control': basic_contrasts['pumps_demean'] - basic_contrasts['control_pumps_demean'],
        'control-pumps': -basic_contrasts['control_pumps_demean'] + basic_contrasts['pumps_demean'],
        'pumps-baseline': basic_contrasts['pumps_demean'],
        'cash-baseline': basic_contrasts['cash_demean'],
        'explode-baseline': basic_contrasts['explode_demean']
        }
    
    contrast_plot = []
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Plot Contrast % 2i out of %i: %s' % (
            index + 1, len(contrasts), contrast_id))
        contrast_plot_path = os.path.join(workflow_out_dir, 'sub-%s_firstlevel_contrast-%s.jpg' % (subj_id+1, contrast_id))
        plot_contrast_matrix(contrast_val, design_matrix, output_file=contrast_plot_path)
        contrast_plot.append(contrast_plot_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return contrasts, contrast_plot
```

### Fit the first level GLM with fixed-effects

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'subj_id': int,
        'subj_imgs': MultiInputFile,
        'subj_masks': MultiInputFile,
        'smoothing_fwhm': float,
        'design_matrices': list,
        'contrasts':dict,
        'return': {'first_level_model': ty.Any, 'z_map_path_dict': dict},
    }
)
def firstlevel_estimation(subj_id, subj_imgs, subj_masks, smoothing_fwhm, design_matrices, contrasts):
    t1 = datetime.datetime.now()
    print(f"\nStart firstlevel estimation for subject-{subj_id+1} ...\n")
    
    import nibabel as nib
    from nilearn.image import math_img
    from nilearn.glm.first_level import FirstLevelModel
    
    print('Compute firstlevel mask...')
    # average mask across three runs
    mean_mask = math_img('np.mean(img, axis=-1)', img=subj_masks)
    # binarize the mean mask
    mask = math_img('img > 0', img=mean_mask)
    # fit the (fixed-effects) firstlevel model with three runs simultaneously
    first_level_model = FirstLevelModel(mask_img=mask, smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
    first_level_model = first_level_model.fit(subj_imgs, design_matrices=design_matrices)
    
    print('Computing contrasts...')
    z_map_path_dict = dict.fromkeys(contrasts.keys())
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(contrasts), contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the two sessions
        z_map = first_level_model.compute_contrast(
            contrast_val, output_type='z_score')

        # write the resulting stat images to file
        z_map_path = os.path.join(workflow_out_dir, 'sub-%s_contrast-%s_z_map.nii.gz' % (subj_id+1, contrast_id))
        z_map_path_dict[contrast_id] = z_map_path
        z_map.to_filename(z_map_path)
    
    t2 = datetime.datetime.now()
    print(t2-t1)
    return first_level_model, z_map_path_dict
```

### Get cluster table and glm report

For publication purposes, we obtain a cluster table and a summary report.

```{code-cell} ipython3
# get cluster table 
@pydra.mark.task
@pydra.mark.annotate(
    {'subj_id': int, 'z_map_path': str, 'return': {'output_file': str}}
)
def cluster_table(subj_id, z_map_path):
    
    import nibabel as nib
    from nilearn.reporting import get_clusters_table
    from scipy.stats import norm

    stat_img = nib.load(z_map_path)
    output_file = os.path.join(workflow_out_dir, 'sub-%s_cluster_table.csv' % subj_id+1)
    df = get_clusters_table(
        stat_img, stat_threshold=norm.isf(0.001), cluster_threshold=10
    )
    df.to_csv(output_file, index=None)
    return output_file

# get glm report
@pydra.mark.task
@pydra.mark.annotate(
    {'subj_id': int, 'model': ty.Any, 'contrasts': ty.Any, 'return': {'output_file': str}}
)
def glm_report(subj_id, model, contrasts):
    from nilearn.reporting import make_glm_report

    output_file = os.path.join(workflow_out_dir, 'sub-%s_glm_report.html' % subj_id+1)
    report = make_glm_report(model, contrasts)
    report.save_as_html(output_file)
    return output_file
```

### Create the first-level GLM workflow

```{code-cell} ipython3
# initiate the first-level GLM workflow
wf_firstlevel = Workflow(
    name='wf_firstlevel',
    input_spec=[
        'subj_id',
        'n_run',
        'tr',
        'n_scans',
        'hrf_model',
        'event_list', 
        'img_list', 
        'mask_list',
        'smoothing_fwhm',
        'output_dir'
    ],
)

wf_firstlevel.split('subj_id')
# add task - get_subj_file
wf_firstlevel.add(
    get_subj_file(
        name = "get_subj_file",
        subj_id = wf_firstlevel.lzin.subj_id, 
        n_run = wf_firstlevel.lzin.n_run, 
        event_list = wf_firstlevel.lzin.event_list, 
        img_list = wf_firstlevel.lzin.img_list, 
        mask_list = wf_firstlevel.lzin.mask_list
    )
)

# add task - get_firstlevel_dm
wf_firstlevel.add(
    get_firstlevel_dm(
        name = "get_firstlevel_dm",
        tr = wf_firstlevel.lzin.tr, 
        n_scans = wf_firstlevel.lzin.n_scans, 
        hrf_model = wf_firstlevel.lzin.hrf_model, 
        subj_id = wf_firstlevel.get_subj_file.lzout.subj_id, 
        subj_imgs = wf_firstlevel.get_subj_file.lzout.subj_imgs, 
        subj_events = wf_firstlevel.get_subj_file.lzout.subj_events, 
    )
)

# add task - set_contrast
wf_firstlevel.add(
    set_contrast(
        name = "set_contrast",
        subj_id = wf_firstlevel.get_subj_file.lzout.subj_id,
        design_matrices = wf_firstlevel.get_firstlevel_dm.lzout.design_matrices
    )
)

# add task - firstlevel_estimation
wf_firstlevel.add(
    firstlevel_estimation(
        name = "firstlevel_estimation",
        subj_id = wf_firstlevel.get_subj_file.lzout.subj_id, 
        subj_imgs = wf_firstlevel.get_subj_file.lzout.subj_imgs, 
        subj_masks = wf_firstlevel.get_subj_file.lzout.subj_masks, 
        smoothing_fwhm = wf_firstlevel.lzin.smoothing_fwhm, 
        design_matrices = wf_firstlevel.get_firstlevel_dm.lzout.design_matrices, 
        contrasts = wf_firstlevel.set_contrast.lzout.contrasts
    )
)

wf_firstlevel.combine('subj_id')
# specify output
wf_firstlevel.set_output(
    [
        ('first_level_designmatrices', wf_firstlevel.get_firstlevel_dm.lzout.design_matrices),
        ('first_level_dm_paths', wf_firstlevel.get_firstlevel_dm.lzout.dm_paths),
        ('first_level_contrast', wf_firstlevel.set_contrast.lzout.contrasts),
        ('first_level_contrast_plot', wf_firstlevel.set_contrast.lzout.contrast_plot),
        ('first_level_model_list', wf_firstlevel.firstlevel_estimation.lzout.first_level_model),
        ('first_level_z_map_dict_list', wf_firstlevel.firstlevel_estimation.lzout.z_map_path_dict),
    ]
)
```

## Second-Level GLM

The second-level estimation contains the following steps:
- construct design matrix
- fit the second-level GLM
- thresholding & cluster analysis

+++ {"tags": []}

### Get second-level design matrix

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {'n_subj': int, 'return': {'design_matrix': ty.Any}}
)
def get_secondlevel_dm(n_subj):
    t1 = datetime.datetime.now()
    print(f"\nGet secondlevel design matrix ...\n")
    import pandas as pd
    design_matrix = pd.DataFrame([1] * n_subj,columns=['intercept'])
    dm_path = os.path.join(workflow_out_dir, 'secondlevel_designmatrix.csv')
    design_matrix.to_csv(dm_path, index=None)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return design_matrix
```

### Fit the second level GLM

Here, we use the list of fitted FirstLevelModel objects as the input for the SecondLevelModel, since all subjects share a similar design matrix (same variables reflected in column names).

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {'second_level_input': ty.Any, 'design_matrix': ty.Any, 'firstlevel_contrast':list, 
     'return': {'secondlevel_mask': ty.Any, 'stat_maps_dict': dict}}
)
def secondlevel_estimation(second_level_input, design_matrix, firstlevel_contrast):
    """ task to estimate the second level
    Parameters
    ----------
    second_level_input : list
        the list of FirstLevelModel
    design_matrix : ty.Any
        a pandas.DataFrame that specifies the second level design
    firstlevel_contrast : dict
        a dictionary of contrasts

    Returns
    -------
    secondlevel_mask : mask from SecondLevelModel
        
    stat_maps_dict : dict
        
    """
    t1 = datetime.datetime.now()
    print(f"\nStart secondlevel estimation ...\n")
    from nilearn.glm.second_level import SecondLevelModel
    second_level_model = SecondLevelModel()
    second_level_model.fit(second_level_input, design_matrix=design_matrix)
    secondlevel_mask = second_level_model.masker_.mask_img_
    print('Computing contrasts...')
    stat_maps_dict = {}
    for index, (contrast_id, contrast_val) in enumerate(firstlevel_contrast[0].items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(firstlevel_contrast[0]), contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the two sessions
        stat_maps = second_level_model.compute_contrast(first_level_contrast=contrast_val, output_type='all')
        stat_maps_dict[contrast_id] = stat_maps
        # # write the resulting stat images to file
        # z_image_path = path.join(output_dir, 'contrast-%s_z_map.nii.gz' % contrast_id)
        # z_image_path_list.append(z_image_path)
        # z_map.to_filename(z_image_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return secondlevel_mask, stat_maps_dict
```

### Cluster-thresholding and Plot without multiple comparison

Threshold the resulting map without multiple comparisons correction, abs(z) > 3.29 (equivalent to p < 0.001), cluster size > 10 voxels.

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {'stat_maps_dict': dict, 'threshold': float, 'cluster_threshold': int, 
     'return': {'thresholded_map_dict': dict, 'plot_contrast_dict': dict}}
)
def cluster_thresholding(stat_maps_dict, threshold, cluster_threshold):
    t1 = datetime.datetime.now()
    print(f"\nStart cluster thresholding ...\n")
    from nilearn.image import threshold_img
    from nilearn import plotting
    thresholded_map_dict = dict.fromkeys(stat_maps_dict.keys())
    plot_contrast_dict = dict.fromkeys(stat_maps_dict.keys())
    for index, (stats_id, stats_val) in enumerate(stat_maps_dict.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(stat_maps_dict), stats_id))
        thresholded_map = threshold_img(
            img = stats_val['z_score'],
            threshold=threshold,
            cluster_threshold=cluster_threshold,
            two_sided=True,
        )
        thresholded_map_path = path.join(workflow_out_dir, 'secondlevel_cluster_thresholded_contrast-%s_z_map.nii.gz' % stats_id)
        thresholded_map_dict[stats_id] = thresholded_map_path
        thresholded_map.to_filename(thresholded_map_path)
        plot_path = os.path.join(workflow_out_dir, 
                                   'secondlevel_cluster_thresholded_contrast-%s_zmap.jpg' % stats_id)
        plot_contrast_dict[stats_id] = plot_path
        plotting.plot_stat_map(thresholded_map, cut_coords=[0],
                               title='Cluster Thresholded z map',
                               output_file=plot_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return thresholded_map_dict, plot_contrast_dict
```

### Multiple comparison and Plot

We have the following choices:
- `fdr`: False Discovery Rate (FDR <.05) and no cluster-level threshold
- `fpr`: False Positive Rate
- `bonferroni`

More details see [here](https://nilearn.github.io/stable/modules/generated/nilearn.glm.threshold_stats_img.html#nilearn.glm.threshold_stats_img)

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {'stat_maps_dict': dict, 'alpha': float, 'height_control': str, 
     'return': {'thresholded_map_dict': dict, 'plot_contrast_dict': dict}}
)
def multiple_comparison(stat_maps_dict, alpha, height_control):
    t1 = datetime.datetime.now()
    print(f"\nStart multiple comparison ...\n")
    from nilearn.glm import threshold_stats_img
    from nilearn import plotting
    thresholded_map_dict = dict.fromkeys(stat_maps_dict.keys())
    plot_contrast_dict = dict.fromkeys(stat_maps_dict.keys())
    for index, (stats_id, stats_val) in enumerate(stat_maps_dict.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(stat_maps_dict), stats_id))
        thresholded_map, threshold = threshold_stats_img(
            stat_img=stats_val['z_score'], 
            alpha=alpha, 
            height_control=height_control)
        thresholded_map_path = os.path.join(workflow_out_dir, 
                                         'secondlevel_multiple_comp_corrected_contrast-%s_z_map.nii.gz' % stats_id)
        thresholded_map_dict[stats_id] = thresholded_map_path
        thresholded_map.to_filename(thresholded_map_path)
        plot_path = os.path.join(workflow_out_dir, 
                                   'secondlevel_multiple_comp_corrected_contrast-%s_zmap.jpg' % stats_id)
        plot_contrast_dict[stats_id] = plot_path
        plotting.plot_stat_map(thresholded_ma,
                               title='Thresholded z map, expected fdr = .05',
                               threshold=threshold, 
                               output_file=plot_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return thresholded_map_dict, plot_contrast_dict
```

### Paramatric test & Plot

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {'stat_maps_dict': list, 
     'second_level_model': ty.Any,
     'return': {'thresholded_map_dict': dict, 'plot_contrast_dict': dict}}
)
def parametric_test(stat_maps_dict, second_level_model):
    t1 = datetime.datetime.now()
    print(f"\nStart parametric test ...\n")
    import numpy as np
    from nilearn.image import get_data, math_img
    from nilearn import plotting
    thresholded_map_dict = dict.fromkeys(stat_maps_dict.keys())
    plot_contrast_dict = dict.fromkeys(stat_maps_dict.keys())
    for index, (stats_id, stats_val) in enumerate(stat_maps_dict.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(stat_maps_dict), stats_id))
        p_val = stats_val['p_value']
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))"
                                .format(str(n_voxels)),
                                img=p_val)
        
        thresholded_map_path = os.path.join(workflow_out_dir, 'secondlevel_paramatric_thresholded_contrast-%s_z_map.nii.gz' % stats_id)
        thresholded_map_dict[stats_id] = thresholded_map_path
        neg_log_pval.to_filename(thresholded_map_path)
    
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there is
        # less than 10% probability to make a single false discovery (90% chance that
        # we make no false discovery at all).  This threshold is much more conservative
        # than the previous one.
        title = ('parametric test (FWER < 10%)')
        plot_path = os.path.join(workflow_out_dir, 
                                   'secondlevel_paramatric_thresholded_contrast-%s_zmap.jpg' % stats_id)
        plot_contrast_dict[stats_id] = plot_path
        plotting.plot_glass_brain(
            neg_log_pval, colorbar=True, display_mode='z', plot_abs=False, 
            vmax=3, threshold=1, title=title, output_file=plot_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    return thresholded_map_dict, plot_contrast_dict
```

### Non-paramatric test & Plot

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {'second_level_input': list,'design_matrix': ty.Any, 'firstlevel_contrast': list, 'n_perm': int, 
     'return': {'thresholded_map_dict': dict, 'plot_contrast_dict': dict}}
)
def nonparametric_test(second_level_input, smoothing_fwhm, design_matrix, firstlevel_contrast, n_perm):
    """ task to estimate the second level
    Parameters
    ----------
    second_level_input : list
        the list of first-level output (dictionary)
    design_matrix : ty.Any
        a pandas.DataFrame that specifies the second level design
    firstlevel_contrast : dict
        a dictionary of contrasts used in the first level
    n_perm: int
        number of permutation

    Returns
    -------
    thresholded_map_dict : dict
        
    plot_contrast_dict : dict
        
    """
    t1 = datetime.datetime.now()
    print(f"\nStart nonparametric test ...\n")
    from nilearn.glm.second_level import non_parametric_inference
    from nilearn import plotting
    thresholded_map_dict = dict.fromkeys(firstlevel_contrast[0].keys())
    plot_contrast_dict = dict.fromkeys(firstlevel_contrast[0].keys())
    for index, (contrast_id, contrast_val) in enumerate(firstlevel_contrast[0].items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(firstlevel_contrast[0]), contrast_id))
        # here we set threshold as none to do voxel-level FWER-correction.
        neg_log_pvals_permuted_ols_unmasked = \
            non_parametric_inference(second_level_input=second_level_input, design_matrix=design_matrix,
                                     model_intercept=True, n_perm=n_perm,first_level_contrast=contrast_val,
                                     two_sided_test=False, smoothing_fwhm=smoothing_fwhm, n_jobs=1)
        print("test1...")
        thresholded_map_path = os.path.join(workflow_out_dir, 'secondlevel_permutation_contrast-%s_z_map.nii.gz' % contrast_id)
        print("test2...")
        thresholded_map_dict[contrast_id] = thresholded_map_path
        print("test3...")
        neg_log_pvals_permuted_ols_unmasked.to_filename(thresholded_map_path)
        # here I actually have more than one contrast
        title = ('permutation test (FWER < 10%)')
        plot_path = os.path.join(workflow_out_dir, 'secondlevel_permutation_contrast-%s_zmap.jpg' % contrast_id)
        plot_contrast_dict[contrast_id] = plot_path
        display = plotting.plot_glass_brain(
            neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
            display_mode='z', plot_abs=False, threshold=1, 
            title=title, output_file=plot_path)
    t2 = datetime.datetime.now()
    print(t2-t1)
    print(f"thresholded_map_dict = {thresholded_map_dict}")
    print(f"plot_contrast_dict = {plot_contrast_dict}")
    return thresholded_map_dict, plot_contrast_dict
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'test_input1':ty.Any,
        'test_input2': ty.Any,
        'return': {'out1':ty.Any, 'out2':ty.Any}
    }
)
def test1(test_input1, test_input2):
    print("testing...")
    out1 = test_input1
    out2 = test_input2
    return out1, out2
```

```{code-cell} ipython3
@pydra.mark.task
@pydra.mark.annotate(
    {
        'test_input1':ty.Any,
        'test_input2': ty.Any,
        'return': {'out1':ty.Any, 'out2':ty.Any}
    }
)
def test2(test_input1, test_input2):
    print("testing...")
    out1 = test_input1
    out2 = test_input2
    return out1, out2
```

### Create the second-level GLM workflow

```{code-cell} ipython3
# initiate the first-level GLM workflow
wf_secondlevel = Workflow(
    name='wf_secondlevel',
    input_spec=[
        'n_subj',
        'second_level_input', 
        'smoothing_fwhm',
        'firstlevel_contrast',
        'n_perm',
        'output_dir'
    ],
)

# add task - get_secondlevel_dm
wf_secondlevel.add(
    get_secondlevel_dm(
        name = "get_secondlevel_dm",
        n_subj = wf_secondlevel.lzin.n_subj, 
    )
)

# # add task - secondlevel_estimation
# wf_secondlevel.add(
#     secondlevel_estimation(
#         name = "secondlevel_estimation",
#         second_level_input = wf_secondlevel.lzin.second_level_input,  
#         design_matrix = wf_secondlevel.get_secondlevel_dm.lzout.design_matrix, 
#         firstlevel_contrast = wf_secondlevel.lzin.firstlevel_contrast
#     )
# )

# # add task - secondlevel_estimation
# wf_secondlevel.add(
#     cluster_thresholding(
#         name = "cluster_thresholding",
#         stat_maps_dict = wf_secondlevel.secondlevel_estimation.lzout.stat_maps_dict, 
#         threshold = 3.29, 
#         cluster_threshold = 10
#     )
# )

# # add task - multiple_comparison
# wf_secondlevel.add(
#     multiple_comparison(
#         name = "multiple_comparison",
#         stat_maps_dict = wf_secondlevel.secondlevel_estimation.lzout.stat_maps_dict, 
#         alpha = 0.05,
#         height_control = 'fdr'
#     )
# )

# # add task - parametric_test
# wf_secondlevel.add(
#     parametric_test(
#         name = "parametric_test",
#         stat_maps_dict = wf_secondlevel.secondlevel_estimation.lzout.stat_maps_dict, 
#         second_level_model = wf_secondlevel.secondlevel_estimation.lzout.second_level_model
#     )
    
# )

# add task - nonparametric_test
wf_secondlevel.add(
    nonparametric_test(
        name = "nonparametric_test",
        second_level_input = wf_secondlevel.lzin.second_level_input,
        smoothing_fwhm = wf_secondlevel.lzin.smoothing_fwhm, 
        design_matrix = wf_secondlevel.get_secondlevel_dm.lzout.design_matrix, 
        firstlevel_contrast = wf_secondlevel.lzin.firstlevel_contrast, 
        n_perm = wf_secondlevel.lzin.n_perm,
    )
)

# wf_secondlevel.add(
#     test1(
#         name = "test1",
#         test_input1 = wf_secondlevel.get_secondlevel_dm.lzout.design_matrix, 
#         test_input2 = wf_secondlevel.get_secondlevel_dm.lzout.design_matrix)
# )

# wf_secondlevel.add(
#     test2(
#         name = "test2",
#         test_input1 = wf_secondlevel.test1.lzout.out1, 
#         test_input2 = wf_secondlevel.test1.lzout.out2)
# )
# specify output
wf_secondlevel.set_output(
    [
        # # ('second_level_clusterthresholding_result', wf_secondlevel.cluster_thresholding.lzout.thresholded_map_dict),
        # ('second_level_clusterthresholding_plot', wf_secondlevel.cluster_thresholding.lzout.plot_contrast_dict),
        # ('second_level_mc_result', wf_secondlevel.multiple_comparison.lzout.thresholded_map_dict),
        # ('second_level_mc_plot', wf_secondlevel.multiple_comparison.lzout.plot_contrast_dict),
        # ('second_level_parametric_test', wf_secondlevel.parametric_test.lzout.thresholded_map_dict),
        # ('second_level_parametric_plot', wf_secondlevel.parametric_test.lzout.plot_contrast_dict),
        ('second_level_nonparametric_test', wf_secondlevel.nonparametric_test.lzout.thresholded_map_dict),
        ('second_level_nonparametric_plot', wf_secondlevel.nonparametric_test.lzout.plot_contrast_dict),
    ]
)
```

## The Ultimate Workflow

Now, let's connect all tasks and workflows together

```{code-cell} ipython3
wf = Workflow(
    name='twolevel_glm',
    input_spec=['subj_id', 'rawdata_url', 'fmriprep_url', 'smoothing_fwhm', 'output_dir'],
)

wf.inputs.rawdata_url = 'https://github.com/OpenNeuroDerivatives/ds000001-fmriprep.git'
wf.inputs.fmriprep_url = 'https://github.com/OpenNeuroDatasets/ds000001.git'
wf.inputs.smoothing_fwhm = 5.0
wf.inputs.output_dir = workflow_out_dir

wf.add(
    get_data(
        name = "get_data",
        rawdata_url = wf.lzin.rawdata_url, 
        fmriprep_url = wf.lzin.fmriprep_url)
)
n_subj = 3
wf_firstlevel.inputs.subj_id = [x for x in range(n_subj)]
wf_firstlevel.inputs.n_run = 3
wf_firstlevel.inputs.tr = 2.3
wf_firstlevel.inputs.n_scans = 300
wf_firstlevel.inputs.hrf_model = 'glover'
wf_firstlevel.inputs.event_list = wf.get_data.lzout.event_list
wf_firstlevel.inputs.img_list = wf.get_data.lzout.img_list
wf_firstlevel.inputs.mask_list = wf.get_data.lzout.mask_list
wf_firstlevel.inputs.smoothing_fwhm = wf.lzin.smoothing_fwhm
wf_firstlevel.inputs.output_dir = wf.lzin.output_dir
wf.add(wf_firstlevel)

wf_secondlevel.inputs.n_subj = n_subj
wf_secondlevel.inputs.second_level_input = wf.wf_firstlevel.lzout.first_level_model_list 
wf_secondlevel.inputs.smoothing_fwhm = wf.lzin.smoothing_fwhm
wf_secondlevel.inputs.firstlevel_contrast = wf.wf_firstlevel.lzout.first_level_contrast
wf_secondlevel.inputs.n_perm = 1
wf_secondlevel.inputs.output_dir = wf.lzin.output_dir
wf.add(wf_secondlevel)

wf.set_output(
    [
        ('first_level_outputs', wf.wf_firstlevel.lzout.first_level_z_map_dict_list),
        # ('second_level_clusterthresholding_result', wf.wf_secondlevel.lzout.second_level_clusterthresholding_result),
        # ('second_level_clusterthresholding_plot', wf.wf_secondlevel.lzout.second_level_clusterthresholding_plot),
        # ('second_level_mc_result', wf.wf_secondlevel.lzout.second_level_mc_result),
        # ('second_level_mc_plot', wf.wf_secondlevel.lzout.second_level_mc_plot),
        # ('second_level_parametric_test', wf.wf_secondlevel.lzout.second_level_parametric_test),
        # ('second_level_parametric_plot', wf.wf_secondlevel.lzout.second_level_parametric_plot),
        ('second_level_nonparametric_test', wf.wf_secondlevel.lzout.second_level_nonparametric_test),
        ('second_level_nonparametric_plot', wf.wf_secondlevel.lzout.second_level_nonparametric_plot),    
    ]
)
```

```{code-cell} ipython3
:tags: []

from pydra import Submitter

with Submitter(plugin='cf', n_procs=8) as submitter:
    submitter(wf)

results = wf.result()

print(results)
```

```{code-cell} ipython3

```
