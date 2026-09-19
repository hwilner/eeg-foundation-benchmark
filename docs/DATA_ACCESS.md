# Data access: TUH EEG Corpus (TUEG and task derivatives)

The benchmark targets the Temple University Hospital (TUH) EEG Corpus family:

| Corpus | Content | Use in this project |
|---|---|---|
| TUEG | Full unlabeled clinical EEG archive (~TB scale) | SSL pretraining |
| TUAB | Abnormal vs. normal classification | Downstream benchmark |
| TUSZ | Seizure detection | Downstream benchmark |
| TUEV | Event classification | Downstream benchmark |

## How to obtain access

The data is **free for research use but requires a signed data use
agreement (DUA) with Temple University**. The data cannot be redistributed;
every contributor must obtain their own access.

1. Fill out the request form at
   https://isip.piconepress.com/projects/tuh_eeg/ (TUH EEG Corpus access
   request; includes agreement terms and contact details).
2. Email the signed form to the address listed on that page (per the
   instructions on the project site).
3. Wait for approval — turnaround can take **days to weeks**. Approved
   users receive credentials for `rsync`-based download from the transfer
   host indicated in the approval email.
4. Plan storage: TUEG is ~TB-scale; the task subsets (TUAB/TUSZ/TUEV) are
   much smaller. Record the corpus version received.

## Compliance notes

- Use is permitted for research (see the DUA for the exact scope and any
  commercial-use provisions); do not share credentials or raw data.
- Do not commit corpus data, file lists beyond public documentation, or
  derived artifacts to this repository.
- Cite the TUH EEG Corpus in publications per the DUA.

## Current pipeline status

The full pipeline in `src/eegfm/` (window dataset interface, channel-set
standardization, transformer encoder, masked-reconstruction pretraining,
linear-probe/fine-tune evaluation, benchmark runner) is **ready for the
corpus once access is granted**. Until then, all development and CI run on
the synthetic EEG generator (`eegfm.simulate`), which plants
class-discriminative cross-channel patterns for data-free validation.

Access progress is tracked in the backlog (see the TUEG license issue).
