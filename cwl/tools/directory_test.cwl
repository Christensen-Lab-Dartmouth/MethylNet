class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: directory_test
baseCommand: []
inputs: []
outputs:
  - id: output
    type: Directory
    outputBinding:
      glob: output_dir/
label: directory_test
arguments:
  - position: 0
    prefix: ''
    shellQuote: false
    valueFrom: mkdir output_dir
requirements:
  - class: ShellCommandRequirement
  - class: DockerRequirement
    dockerPull: 'alpine:latest'
