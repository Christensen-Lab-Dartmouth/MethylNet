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
outputs:
  - id: output_sample_sheet
    type: File
    outputBinding:
      glob: $(inputs.input_sample_sheet.path)
  - id: idat_dir
    type: Directory
    outputBinding:
      glob: '$(inputs.input_sample_sheet.path.split(''/'').slice(0,-1).join(''/''))'
label: meffil_encode
arguments:
  - position: 0
    prefix: python /scripts/preprocess.py meffil_encode -os
    shellQuote: false
    valueFrom: $(inputs.input_sample_sheet.path)
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 2000
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
  - class: InlineJavascriptRequirement
