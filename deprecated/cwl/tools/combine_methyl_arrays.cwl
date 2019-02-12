class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: combine_methyl_arrays
baseCommand: []
inputs:
  - id: input_pkls
    type: 'File[]'
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            cmd=""
            for (var i=0; i < inputs.input_pkls.length;i++){
                cmd += " -i "+inputs.input_pkls[i].path
            }
            return cmd
        }
outputs:
  - id: output_methyl_array
    type: File
    outputBinding:
      glob: ./preprocessed_output/methyl_array.pkl
label: combine_methyl_arrays
arguments:
  - position: 0
    prefix: python combine_methylation_arrays -o
    shellQuote: false
    valueFrom: ./preprocessed_output/methyl_array.pkl
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 4000
    coresMin: 0
  - class: DockerRequirement
    dockerPull: 'methylnet:dev'
  - class: InlineJavascriptRequirement
