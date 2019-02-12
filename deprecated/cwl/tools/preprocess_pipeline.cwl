class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: preprocess_pipeline
baseCommand: []
inputs:
  - id: idat_csv_dir_input
    type: Directory
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: idat_dir
    type: Directory?
  - id: n_cores
    type: int?
    inputBinding:
      position: 0
      prefix: '-n'
      shellQuote: false
  - id: meffil
    type: boolean?
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            if (inputs.meffil){
                return '-m'
            }
            else {
                return ''
            }
        }
outputs:
  - id: output_pkl
    type: File
    outputBinding:
      glob: ./preprocess_outputs/methyl_array.pkl
label: preprocess_pipeline
arguments:
  - position: 0
    prefix: ln -s
    shellQuote: false
    valueFrom: $(inputs.idat_dir.path) geo_idats/
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
