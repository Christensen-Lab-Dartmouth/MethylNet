class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: download_geo
baseCommand: []
inputs:
  - id: query
    type: string
    inputBinding:
      position: 0
      prefix: '-g'
      shellQuote: false
outputs:
  - id: idat_dir
    type: Directory
    outputBinding:
      glob: geo_idats
  - id: initial_sample_sheet
    type: File
    outputBinding:
      glob: geo_idats/*.csv
label: download_geo
arguments:
  - position: 0
    prefix: python /scripts/preprocess.py download_geo -o
    shellQuote: false
    valueFrom: geo_idats/
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 8000
    coresMin: 0
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
