class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: create_sample_sheet
baseCommand:
  - /scripts/preprocess.py
  - create_sample_sheet
inputs:
  - id: input_sample_sheet
    type: File
    inputBinding:
      position: 0
      prefix: '-is'
      shellQuote: false
  - id: source_type
    type: string
    inputBinding:
      position: 0
      prefix: '-s'
      shellQuote: false
  - id: idat_dir
    type: Directory
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: input
    type: File
    inputBinding:
      position: 0
outputs: []
label: create_sample_sheet
arguments:
  - position: 0
    prefix: '-os'
requirements:
  - class: ShellCommandRequirement
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
