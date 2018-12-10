class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: create_sample_sheet
baseCommand:
  - python /scripts/preprocess.py
  - create_sample_sheet
inputs:
  - id: input_sample_sheet
    type: File
    inputBinding:
      position: 0
      prefix: '-is'
      shellQuote: false
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
      glob: $(inputs.idat_dir.path)
  - id: final_csv
    type: File
    outputBinding:
      glob: $(inputs.idat_dir.path)/$(inputs.input_sample_sheet.nameroot)_final.csv
label: create_sample_sheet
arguments:
  - position: 0
    prefix: '-os'
    shellQuote: false
    valueFrom: $(inputs.idat_dir.path)/$(inputs.input_sample_sheet.nameroot)_final.csv
  - position: 1
    prefix: '&& mkdir backup_sheets && mv'
    shellQuote: false
    valueFrom: $(inputs.input_sample_sheet.path) backup_sheets
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 4000
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
  - class: InlineJavascriptRequirement
