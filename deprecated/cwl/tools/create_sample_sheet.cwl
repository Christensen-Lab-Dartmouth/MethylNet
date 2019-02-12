class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: create_sample_sheet
baseCommand: []
inputs:
  - id: input_sample_sheet
    type: File
    inputBinding:
      position: 0
      prefix: '-is'
      shellQuote: false
      valueFrom: geo_idats/$(inputs.input_sample_sheet.basename)
  - id: source_type
    type: string?
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
      valueFrom: geo_idats/
  - id: mapping_file
    type: File?
    inputBinding:
      position: 0
      prefix: '-m'
      shellQuote: false
  - id: header_line
    type: int?
    inputBinding:
      position: 0
      prefix: '-l'
      shellQuote: false
  - id: disease_class_column
    type: string?
    inputBinding:
      position: 0
      prefix: '-d'
      shellQuote: false
      valueFrom: '''$(inputs.disease_class_column)'''
  - id: base_name_col
    type: string?
    inputBinding:
      position: 0
      prefix: '-b'
      shellQuote: false
      valueFrom: '''$(inputs.base_name_col)'''
  - id: include_columns_file
    type: File?
    inputBinding:
      position: 0
      prefix: '-c'
      shellQuote: false
outputs:
  - id: idat_dir_out
    type: Directory
    outputBinding:
      glob: geo_idats
  - id: final_csv
    type: File
    outputBinding:
      glob: geo_idats/$(inputs.input_sample_sheet.nameroot)_final.csv
label: create_sample_sheet
arguments:
  - position: 0
    prefix: ''
    shellQuote: false
    valueFrom: >-
      mkdir geo_idats && mv $(inputs.idat_dir.path+'/*') geo_idats && python
      /scripts/preprocess.py create_sample_sheet -os
      geo_idats/$(inputs.input_sample_sheet.nameroot)_final.csv
  - position: 1
    prefix: '&& mkdir backup_sheets && mv'
    shellQuote: false
    valueFrom: geo_idats/$(inputs.input_sample_sheet.basename) backup_sheets
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 4000
  - class: DockerRequirement
    dockerPull: 'methylnet:dev'
  - class: InlineJavascriptRequirement
