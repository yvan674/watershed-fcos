# Watershed-FCOS Dev Tools
Tools for the continuing development of the Deep Watershed Detector.

These are the tools used to aid the continuing development of the Deep Watershed Detector.
Amongst the tools provided are ground truth visualizers for the COCO dataset, a script to calculate the mean and std of a dataset, and dataset conversion utilities

## Deepscores Conversion Notes

- Class colors are not changed for online or offline.
    - Only the names are changed.
- Small and not small class_colors are not changed.
    - Only the names are changed.
- Clef symbols and clef changes are converted to just clef symbols by this package.

## Notes
- Make sure when making a new class_names.csv file that category numbers are correct, especially for clefs.