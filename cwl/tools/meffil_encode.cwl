class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: meffil_encode
baseCommand: []
inputs:
  - id: input_sample_sheet
    type: File
    inputBinding:
      position: 0
      prefix: '-is'
      shellQuote: false
      valueFrom: geo_idats/$(inputs.input_sample_sheet.basename)
  - id: idat_dir_input
    type: Directory
outputs:
  - id: output_sample_sheet
    type: File
    outputBinding:
      glob: geo_idats/$(inputs.input_sample_sheet.basename)
  - id: idat_dir
    type: Directory
    outputBinding:
      glob: geo_idats
label: meffil_encode
arguments:
  - position: 0
    prefix: mkdir geo_idats && mv
    shellQuote: false
    valueFrom: >-
      $(inputs.idat_dir_input.path+'/*') geo_idats && python
      /scripts/preprocess.py meffil_encode -os
      geo_idats/$(inputs.input_sample_sheet.basename)
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 2000
  - class: DockerRequirement
    dockerPull: 'methylnet:dev'
  - class: InlineJavascriptRequirement
