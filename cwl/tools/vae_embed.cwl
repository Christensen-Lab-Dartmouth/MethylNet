class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: vae_embed
baseCommand: []
inputs:
  - id: input_pkl
    type: File
    inputBinding:
      position: 0
      prefix: '-i'
      shellQuote: false
  - id: cuda
    type: boolean?
    inputBinding:
      position: 0
      shellQuote: false
      valueFrom: |-
        ${
            if (inputs.cuda) {
                return '-c'
            }
            else {
                return ''
            }
        }
  - id: n_latent
    type: int?
    inputBinding:
      position: 0
      prefix: '-n'
      shellQuote: false
  - id: learning_rate
    type: float?
    inputBinding:
      position: 0
      prefix: '-lr'
      shellQuote: false
  - id: weight_decay
    type: float?
    inputBinding:
      position: 0
      prefix: '-wd'
      shellQuote: false
  - id: n_epochs
    type: int?
    inputBinding:
      position: 0
      prefix: '-e'
      shellQuote: false
  - id: hidden_layer_encoder_topology
    type: string?
    inputBinding:
      position: 0
      prefix: '-hlt'
      shellQuote: false
outputs:
  - id: output_methyl_array_encoded
    type: File
    outputBinding:
      glob: ./embeddings/*.pkl
  - id: pytorch_model
    type: File?
    outputBinding:
      glob: ./embeddings/output_model.p
label: vae_embed
arguments:
  - position: 0
    prefix: python /scripts/embedding.py perform_embedding -o
    shellQuote: false
    valueFrom: ./embeddings/
requirements:
  - class: ShellCommandRequirement
  - class: ResourceRequirement
    ramMin: 8000
    coresMin: 0
  - class: DockerRequirement
    dockerPull: 'methylnet:0.1'
  - class: InlineJavascriptRequirement
