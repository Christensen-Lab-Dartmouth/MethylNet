class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: imputation
baseCommand: []
inputs:
  - id: input_pkl
    type: File
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: imputation_method
    type: string?
    inputBinding:
      position: 0
      prefix: '-m'
      shellQuote: false
  - id: solver
    type: string?
    inputBinding:
      position: 0
      prefix: '-s'
      shellQuote: false
  - id: n_neighbors
    type: int?
    inputBinding:
      position: 0
      prefix: '-n'
      shellQuote: false
outputs:
  - id: imputed_methylarray
    type: File
    outputBinding:
      glob: ./imputed_outputs/methyl_array.pkl
label: imputation
arguments:
  - position: 0
    prefix: python /scripts/preprocess.py imputation_pipeline -o
    shellQuote: false
    valueFrom: ./imputed_outputs/methyl_array.pkl
requirements:
  - class: ShellCommandRequirement
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
