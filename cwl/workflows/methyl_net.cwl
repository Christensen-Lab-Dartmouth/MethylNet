class: Workflow
cwlVersion: v1.0
id: methyl_net
label: methyl_net
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: query
    type: string
    'sbg:x': -108.6614761352539
    'sbg:y': -35.01996612548828
  - id: include_columns_file
    type: File?
    'sbg:x': 89.87889099121094
    'sbg:y': -74.16876983642578
  - id: disease_class_column
    type: string?
    'sbg:x': 95.18537139892578
    'sbg:y': 165.68411254882812
  - id: n_cores
    type: int
    'sbg:x': 671.5182495117188
    'sbg:y': -99.33586120605469
  - id: split_by_subtype
    type: boolean?
    'sbg:x': 592.386474609375
    'sbg:y': -261
  - id: disease_only
    type: boolean?
    'sbg:x': 417.051025390625
    'sbg:y': -205.59689331054688
  - id: subtype_delimiter
    type: string?
    'sbg:exposed': true
  - id: imputation_method
    type: string?
    'sbg:exposed': true
  - id: solver
    type: string?
    'sbg:exposed': true
  - id: n_neighbors
    type: int?
    'sbg:x': 852.6881713867188
    'sbg:y': -156.36669921875
  - id: n_top_cpgs
    type: int?
    'sbg:x': 1103.4071044921875
    'sbg:y': -121.16587829589844
  - id: cuda
    type: boolean?
    'sbg:exposed': true
  - id: n_latent
    type: int?
    'sbg:x': 1326.7222900390625
    'sbg:y': -137.82904052734375
  - id: learning_rate
    type: float?
    'sbg:exposed': true
  - id: weight_decay
    type: float?
    'sbg:exposed': true
  - id: n_epochs
    type: int?
    'sbg:exposed': true
  - id: hidden_layer_encoder_topology
    type: string?
    'sbg:x': 1329.37060546875
    'sbg:y': 267.35101318359375
  - id: column_of_interest
    type: string?
    'sbg:exposed': true
  - id: n_neighbor
    type: int?
    'sbg:x': 1610.08251953125
    'sbg:y': -127.23672485351562
  - id: axes_off
    type: boolean?
    'sbg:exposed': true
  - id: supervised
    type: boolean?
    'sbg:exposed': true
outputs:
  - id: pytorch_model
    outputSource:
      - vae_embed/pytorch_model
    type: File?
    'sbg:x': 1752.6168212890625
    'sbg:y': 113.90625
  - id: idat_dir
    outputSource:
      - meffil_encode/idat_dir
    type: Directory
    'sbg:x': 788.0591430664062
    'sbg:y': 113.90625
  - id: idat_dir_out
    outputSource:
      - create_sample_sheet/idat_dir_out
    type: Directory
    'sbg:x': 467.06390380859375
    'sbg:y': 113.90625
  - id: output_visual
    outputSource:
      - transform_plot/output_visual
    type: File
    'sbg:x': 1972.8907470703125
    'sbg:y': -10.714360237121582
steps:
  - id: download_geo
    in:
      - id: query
        source: query
    out:
      - id: idat_dir
      - id: initial_sample_sheet
    run: ../tools/download_geo.cwl
    label: download_geo
    'sbg:x': 0
    'sbg:y': 53.453125
  - id: create_sample_sheet
    in:
      - id: input_sample_sheet
        source: download_geo/initial_sample_sheet
      - id: source_type
        default: geo
      - id: idat_dir
        source: download_geo/idat_dir
      - id: header_line
        default: 0
      - id: disease_class_column
        source: disease_class_column
      - id: include_columns_file
        source: include_columns_file
    out:
      - id: idat_dir_out
      - id: final_csv
    run: ../tools/create_sample_sheet.cwl
    label: create_sample_sheet
    'sbg:x': 200.4014892578125
    'sbg:y': 53.453125
  - id: meffil_encode
    in:
      - id: input_sample_sheet
        source: create_sample_sheet/final_csv
    out:
      - id: output_sample_sheet
      - id: idat_dir
    run: ../tools/meffil_encode.cwl
    label: meffil_encode
    'sbg:x': 467.06390380859375
    'sbg:y': 0
  - id: preprocess_pipeline
    in:
      - id: idat_dir_csv
        source: meffil_encode/output_sample_sheet
      - id: n_cores
        source: n_cores
      - id: split_by_subtype
        source: split_by_subtype
      - id: disease_only
        source: disease_only
      - id: subtype_delimiter
        default: ','
        source: subtype_delimiter
    out:
      - id: output_pkl
    run: ../tools/preprocess_pipeline.cwl
    label: preprocess_pipeline
    'sbg:x': 788.0591430664062
    'sbg:y': 7
  - id: vae_embed
    in:
      - id: input_pkl
        source: mad_filter/output_pkl
      - id: cuda
        source: cuda
      - id: n_latent
        default: 100
        source: n_latent
      - id: learning_rate
        default: 0.000005
        source: learning_rate
      - id: weight_decay
        default: 0.0001
        source: weight_decay
      - id: n_epochs
        default: 50
        source: n_epochs
      - id: hidden_layer_encoder_topology
        default: '1000,500'
        source: hidden_layer_encoder_topology
    out:
      - id: output_methyl_array_encoded
      - id: pytorch_model
    run: ../tools/vae_embed.cwl
    label: vae_embed
    'sbg:x': 1443.34033203125
    'sbg:y': 53.453125
  - id: imputation
    in:
      - id: input_pkl
        source: preprocess_pipeline/output_pkl
      - id: imputation_method
        default: KNN
        source: imputation_method
      - id: solver
        default: fancyimpute
        source: solver
      - id: n_neighbors
        default: 5
        source: n_neighbors
    out:
      - id: imputed_methylarray
    run: ../tools/imputation.cwl
    label: imputation
    'sbg:x': 998.46533203125
    'sbg:y': 60.453125
  - id: mad_filter
    in:
      - id: input_pkl
        source: imputation/imputed_methylarray
      - id: n_top_cpgs
        default: 200000
        source: n_top_cpgs
    out:
      - id: output_pkl
    run: ../tools/mad_filter.cwl
    label: mad_filter
    'sbg:x': 1248.52783203125
    'sbg:y': 60.453125
  - id: transform_plot
    in:
      - id: input_pkl
        source:
          - vae_embed/output_methyl_array_encoded
          - mad_filter/output_pkl
      - id: column_of_interest
        default: disease
        source: column_of_interest
      - id: n_neighbor
        default: 5
        source: n_neighbor
      - id: axes_off
        source: axes_off
      - id: supervised
        source: supervised
    out:
      - id: output_visual
    run: ../tools/transform_plot.cwl
    label: transform_plot
    scatter:
      - input_pkl
    scatterMethod: dotproduct
    'sbg:x': 1758.0106201171875
    'sbg:y': -8.01528263092041
requirements:
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement
