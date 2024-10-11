# Utilities

## Usage

Most utilities can be ignored as they are called directly by other scripts. The only utility that the user may want to run is [`ADNI_subjects.sh`](./ADNI_subjects.sh).

This script is just run without any arguments using:

```
module load freesurfer
./ADNI_subjects.sh
```

This will copy over ADNI subjects from another directory, convert them all to `nifti` format and sort the files according to the required directory structure. 

If there are ever more ADNI subjects, all the user needs to do is edit the file and add more subject names before the second `EOM` that marks the end of the subject list.