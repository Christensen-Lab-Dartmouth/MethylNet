class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: transform_plot
baseCommand: []
inputs:
  - id: input_pkl
    type: File
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: column_of_interest
    type: string?
    inputBinding:
      position: 0
      prefix: '-c'
      shellQuote: false
  - id: n_neighbor
    type: int?
    inputBinding:
      position: 0
      prefix: '-nn'
      shellQuote: false
  - id: axes_off
    type: boolean?
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            if (inputs.axes_off) {
                return '-a'
            }
            else {
                return ''
            }
        }
  - id: supervised
    type: boolean?
    inputBinding:
      position: 0
      valueFrom: |-
        ${
            if (inputs.supervised) {
                return '-s'
            }
            else {
                return ''
            }
        }
outputs:
  - id: output_visual
    type: File
    outputBinding:
      glob: visualization.html
label: transform_plot
arguments:
  - position: 0
    prefix: python /scripts/visualizations.py transform_plot -o
    shellQuote: false
    valueFrom: visualization.html
requirements:
  - class: ShellCommandRequirement
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
  - class: InlineJavascriptRequirement
