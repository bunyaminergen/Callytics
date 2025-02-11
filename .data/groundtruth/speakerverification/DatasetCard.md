# Callytics Speaker Verification Dataset *(CSVD)*

This dataset, named **Callytics Speaker Verification Dataset** (CSVD), was created to test and develop **Speaker
Verification**
models specifically for the `Callytics` project. It primarily features Turkish speakers and was developed in
collaboration with Turkish student volunteers. The recordings originate from WhatsApp phone calls, which were captured
using the Cube ASR mobile application in AMR format and later converted to WAV using FFmpeg.

---

### Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Metadata](#metadata)
- [Transcript](#transcript)
- [Versioning](#versioning)
- [Upcoming](#Upcoming)
- [License](#licence)
- [Team](#team)
- [Contact](#contact)
- [Citation](#citation)

---

### Features

- **Dataset Name**: **Callytics Speaker Verification Dataset _(CSVD)_**
- **Version**: v1 (Initial)
- **Purpose**: This dataset is created to test and develop **Speaker Verification** models, especially for the
  `Callytics` project.
- **Language / Accent**: Primarily Turkish speakers
- **Contributors**: Created in collaboration with Turkish student volunteers.
- **Recording Source**: WhatsApp calls (originally AMR) captured Cube ASR mobile application and then converted to WAV
  using FFmpeg.

This version (v1) contains **10 samples** of phone calls involving **Customer Service Representative (CSR)** and *
*Customer* roles. The audio files are stored in **WAV** format with the following specifications:

- **Channels**: Mono
- **Format**: WAV
- **Audio Bitrate**: 128 kbps
- **Sample Rate**: 8.00 kHz
- **Conversion**: AMR → WAV

---

### File Structure

```Text
.
└── speakerverification
    ├── call
    │   └── v1
    │       ├── Call01.wav
    │       ├── Call02.wav
    │       └── metadata.json
    ├── DatasetCard.md
    ├── LICENSE
    ├── README.md
    └── voiceprint
        └── v1
            ├── CSR01.wav
            └── metadata.json

6 directories, 7 files
```

- **call**: Contains call samples (Call01.wav, Call02.wav, etc.) plus a metadata.json.
- **voiceprint/v1**: Contains shorter CSR recordings (CSR01.wav, etc.) plus a metadata.json.
- **DatasetCard.md**: This document, explaining dataset details.
- **LICENSE**: License file for the dataset.

---

### Metadata

##### Call Metadata

###### Sample

```json
  {
    "file_id": "Call01.wav",
    "csr_id": "CSR01",
    "duration": 1.13,
    "recorded": "2025-02-11"
  }
```

- **file_id**: Filename of the audio file.
- **csr_id**: CSR identifier.
- **duration**: Duration of the call.
- **recorded**: Recording date.

##### Voiceprint Metadata

###### Sample

```json
  {
    "file_id": "CSR01.wav",
    "csr_id": "CSR01",
    "gender": "F",
    "age": 22,
    "duration": 0.7,
    "transcript": "Hello and welcome to Callytics! My name is Emily. How may I assist you?",
    "recorded": "2025-02-11"
  }
```

- **file_id**: Filename of the audio file.
- **csr_id**: CSR identifier.
- **gender**: Gender of CSR.
- **age**: Age of CSR.
- **duration**: Duration of the Voice Print.
- **recorded**: Recording date.

---

### Transcript

We provide SRT transcripts for each call under the `call/v1` directory. Each `.srt` file contains:

- Speaker labels
- Time stamps in `HH:MM:SS,MMM` format
- Conversation text

Such as:

- `Call01.wav` has a corresponding `Call01.srt`
- `Call02.wav` has a corresponding `Call02.srt`

###### Annotation Process

The timestamps in the transcripts were manually annotated using **Audacity** for precise time labeling.

###### Transcript Sample

```Text
1
00:00:03,453 --> 00:00:09,166
[CSR] Hello and welcome to Callytics! My name is Emily. How may I assist you?

2
00:00:09,582 --> 00:00:20,410
[Customer] Hi, Emily. I’ve been having trouble receiving notification emails.
They used to arrive instantly, but now there’s a long delay.

3
00:00:20,599 --> 00:00:27,879
[CSR] I’m sorry for the inconvenience. Let me take a look.
Could you give me your account details so I can verify the email settings?
```

---

### Versioning

- `v1 (Initial)`

---

### Upcoming

- [ ] **Potentially larger sets of calls**
- [ ] **Inclusion of new languages or dialects**
- [ ] **Measure Audio Data Quality**
    - *General Data Quality Indicators*
        - [ ] Completeness
        - [ ] Accuracy
        - [ ] Consistency
        - [ ] Uniqueness (Duplicate Check)
        - [ ] Coverage / Diversity
        - [ ] Timeliness
        - [ ] Class Balance and Metadata Richness
    - *Audio-Specific Quality Indicators*
        - [ ] Signal-to-Noise Ratio (SNR)
        - [ ] Distortion Measures
        - [ ] Clipping Rate (Clipped Samples)
        - [ ] Silence Ratio
        - [ ] Noise Types and Levels
        - [ ] Reverberation (Echo) Ratio / DRR (Direct to Reverberant Ratio)
        - [ ] Speaker / Content Distribution
        - [ ] Transcription Quality

---

### Licence

- [LICENSE](LICENSE)

---

### Team

- [Bunyamin Ergen](https://www.linkedin.com/in/bunyaminergen)
- **Turkish Volunteers**

---

### Contact

- [Mail](mailto:info@bunyaminergen.com)

---

### Citation

```bibtex
@software{       CSVD,
  author       = {Bunyamin Ergen},
  title        = {Callytics Speaker Verification Dataset},
  year         = {2025},
  month        = {02},
  url          = {https://github.com/bunyaminergen/Callytics},
  version      = {v1},
}
```

---
