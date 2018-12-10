class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: mad_filter
baseCommand:
  - python /scripts/preprocess.py
  - mad_filter
inputs:
  - id: input_pkl
    type: File
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: n_top_cpgs
    type: int?
    inputBinding:
      position: 0
      prefix: '-n'
      shellQuote: false
outputs:
  - id: output_pkl
    type: File
    outputBinding:
      glob: ./final_preprocessed/methyl_array.pkl
label: mad_filter
arguments:
  - position: 0
    prefix: '-o'
    shellQuote: false
    valueFrom: ./final_preprocessed/methyl_array.pkl
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 2000
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
