class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: split_by_subtype
baseCommand: []
inputs:
  - id: idat_csv
    type: File
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: disease_only
    type: boolean?
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            if (inputs.disease_only) {
                return "-d"
            }
            else {
                return ''
            }
        }
  - id: subtype_delimiter
    type: string?
    inputBinding:
      position: 0
      prefix: '-s'
      shellQuote: false
      valueFrom: '''$(inputs.subtype_delimiter)'''
outputs:
  - id: output_dirs
    type: 'Directory[]'
    outputBinding:
      glob: ./preprocess_outputs/*
label: split_by_subtype
arguments:
  - position: 0
    prefix: python split_preprocess_input_by_subtype -o
    shellQuote: false
    valueFrom: ./preprocess_outputs/
requirements:
  - class: ShellCommandRequirement
  - class: DockerRequirement
    dockerPull: 'methylnet:dev'
  - class: InlineJavascriptRequirement
