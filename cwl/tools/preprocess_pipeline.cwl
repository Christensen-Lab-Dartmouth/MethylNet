class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: preprocess_pipeline
baseCommand: []
inputs:
  - id: idat_dir_csv
    type: File
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
      valueFrom: geo_idats/$(inputs.idat_dir_csv.basename)
  - id: n_cores
    type: int
    inputBinding:
      position: 0
      prefix: '-n'
      shellQuote: false
  - id: split_by_subtype
    type: boolean?
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            if (inputs.split_by_subtype) {
                return '-ss'
            }
            else {
                return ''
            }
        }
  - id: disease_only
    type: boolean?
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            if (inputs.disease_only) {
                return '-d'
            }
            else {
                return ''
            }
        }
  - id: subtype_delimiter
    type: string?
    inputBinding:
      position: 0
      prefix: '-sd'
      shellQuote: false
      valueFrom: '''$(inputs.subtype_delimiter)'''
  - id: idat_dir
    type: Directory?
outputs:
  - id: output_pkl
    type: File
    outputBinding:
      glob: ./preprocess_outputs/methyl_array.pkl
label: preprocess_pipeline
arguments:
  - position: 0
    prefix: mkdir geo_idats && mv
    shellQuote: false
    valueFrom: $(inputs.idat_dir.path+'/*') geo_idats
  - position: 0
    prefix: '&& python /scripts/preprocess.py preprocess_pipeline -m -o'
    shellQuote: false
    valueFrom: ./preprocess_outputs/methyl_array.pkl
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 8000
    coresMin: |-
      ${
          if (inputs.n_cores){
          return inputs.n_cores+1
          }
          else {
              return 1
          }
      }
  - class: DockerRequirement
    dockerPull: 'methylnet:dev'
  - class: InlineJavascriptRequirement
